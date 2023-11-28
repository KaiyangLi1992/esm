from datetime import datetime, timedelta
import os
import subprocess
import ipdb
def kill_process(pid):
    try:
        # 使用 'kill' 命令杀掉进程
        subprocess.run(['kill', pid], check=True)
        print(f"Successfully killed process with PID: {pid}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to kill process {pid}: {e}")

def get_pid_from_files_and_kill(start_time_str, end_time_str, directory):
    start_time = datetime.strptime(start_time_str, "%Y-%m-%d_%H-%M-%S")
    end_time = datetime.strptime(end_time_str, "%Y-%m-%d_%H-%M-%S")
    current_time = start_time

    while current_time <= end_time:
        file_name = f"{current_time.strftime('%Y-%m-%d_%H-%M-%S')}.log"
        file_path = os.path.join(directory, file_name)

        try:
            with open(file_path, 'r') as file:
                for line in file:
                    if 'pid:' in line:
                        pid = line.split('pid:')[1].strip()
                        kill_process(pid)
                        break
        except FileNotFoundError:
            pass

        current_time += timedelta(seconds=1)

# 示例用法
directory = './log' # 替换为您的日志文件所在目录
start_time_str = "2023-11-24_12-12-04"
end_time_str = "2023-11-24_12-18-33"
get_pid_from_files_and_kill(start_time_str, end_time_str, directory)

