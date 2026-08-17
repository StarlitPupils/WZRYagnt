import subprocess
cmd = 'adb -s d1cbbc0f shell input tap 720 1600'
print(f"执行: {cmd}")
subprocess.call(cmd, shell=True)
