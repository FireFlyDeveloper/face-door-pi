#!/bin/bash
cd /home/admin/face-door-system
pkill -f "python.*main.py" 2>/dev/null
export PYTHONUNBUFFERED=1
nohup python main.py --headless > /tmp/face-door.log 2>&1 &
echo $!
