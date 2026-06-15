import os
import subprocess

command = "echo hello world"
r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
print(r.stdout)
print(r.stderr)