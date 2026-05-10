"""
Bluetooth SPP server for face recognition door system.
Uses PyBluez (python3-bluez) for RFCOMM communication.
Protocol: each message is a complete JSON object terminated by newline.
"""

import json
import socket
import select

try:
    import bluetooth
except ImportError:
    bluetooth = None
    print("[BluetoothServer] WARNING: PyBluez not available, bluetooth disabled")


class BluetoothServer:
    """RFCOMM Bluetooth SPP server for remote door system control."""

    UUID_SPP = "00001101-0000-1000-8000-00805F9B34FB"

    def __init__(self, port=1, backlog=1):
        self.port = port
        self.backlog = backlog
        self.server_sock = None
        self.client_sock = None
        self.client_address = None
        self._running = False

    def start(self):
        """Bind and listen on the RFCOMM socket. Set discoverable. Print MAC + port."""
        if bluetooth is None:
            print("[BluetoothServer] PyBluez not available, cannot start")
            return False
        try:
            self.server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            self.server_sock.bind(("", self.port))
            self.server_sock.listen(self.backlog)
            self._running = True

            # Set discoverable
            try:
                self._set_discoverable()
            except Exception as e:
                print(f"[BluetoothServer] Could not set discoverable: {e}")

            # Get local MAC address
            mac = self._get_mac()
            print(f"[BluetoothServer] Started on {mac}:{self.port} (SPP UUID: {self.UUID_SPP})")
            return True
        except Exception as e:
            print(f"[BluetoothServer] Failed to start: {e}")
            self._running = False
            self._cleanup_socket(self.server_sock)
            self.server_sock = None
            return False

    def stop(self):
        """Close server socket and any active client connection."""
        self._running = False
        self.disconnect_client()
        self._cleanup_socket(self.server_sock)
        self.server_sock = None
        print("[BluetoothServer] Stopped")

    def wait_for_connection(self, timeout=None):
        """Accept a client connection. Returns True on success, False on failure."""
        if bluetooth is None or not self._running or self.server_sock is None:
            return False
        try:
            # Non-blocking accept with timeout via select
            if timeout is not None:
                ready = select.select([self.server_sock], [], [], timeout)
                if not ready[0]:
                    return False

            self.client_sock, self.client_address = self.server_sock.accept()
            print(f"[BluetoothServer] Client connected: {self.client_address}")
            return True
        except Exception as e:
            print(f"[BluetoothServer] Accept failed: {e}")
            return False

    def send(self, data_dict):
        """Send a JSON dict as a single newline-terminated line over client socket."""
        if self.client_sock is None:
            return
        try:
            message = json.dumps(data_dict) + '\n'
            self.client_sock.send(message.encode('utf-8'))
        except Exception as e:
            print(f"[BluetoothServer] Send failed: {e}")

    def receive(self, timeout=30.0):
        """Read one JSON line from the client socket. Returns dict or None on timeout/disconnect."""
        if self.client_sock is None:
            return None
        try:
            ready = select.select([self.client_sock], [], [], timeout)
            if not ready[0]:
                # Timeout — no data available
                return None

            data = self.client_sock.recv(4096)
            if not data:
                print("[BluetoothServer] Client disconnected")
                self.disconnect_client()
                return None

            # Decode and parse the first line of JSON
            text = data.decode('utf-8').strip()
            return json.loads(text)
        except json.JSONDecodeError:
            print(f"[BluetoothServer] Invalid JSON received")
            return None
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            print(f"[BluetoothServer] Connection error during receive: {e}")
            self.disconnect_client()
            return None
        except Exception as e:
            print(f"[BluetoothServer] Receive error: {e}")
            return None

    def close_connection(self):
        """Close the client socket."""
        self._cleanup_socket(self.client_sock)
        self.client_sock = None
        self.client_address = None

    def disconnect_client(self):
        """Close the current client connection."""
        if self.client_sock is not None:
            try:
                self.client_sock.close()
            except Exception:
                pass
            self.client_sock = None
            self.client_address = None
            print("[BluetoothServer] Client disconnected")

    def is_client_connected(self):
        """Check if a client is currently connected."""
        return self.client_sock is not None

    # --- Internal helpers ---

    def _get_mac(self):
        """Get the local Bluetooth MAC address."""
        try:
            return bluetooth.read_local_bdaddr()
        except Exception:
            return "00:00:00:00:00:00"

    def _set_discoverable(self):
        """Try to make the device discoverable via hciconfig."""
        import subprocess
        subprocess.run(
            ["sudo", "hciconfig", "hci0", "piscan"],
            capture_output=True,
            timeout=5
        )

    @staticmethod
    def _cleanup_socket(sock):
        """Safely close a socket if it exists."""
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass
