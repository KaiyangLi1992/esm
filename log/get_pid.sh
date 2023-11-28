start_time="2023-11-24_12-12-04"
end_time="2023-11-24_12-18-33"

# ...

# 调整日期时间格式
start_time_formatted=$(echo $start_time | sed 's/_/ /; s/-/:/g')
end_time_formatted=$(echo $end_time | sed 's/_/ /; s/-/:/g')

# 转换为秒
start_seconds=$(date -d "$start_time_formatted" +%s)
end_seconds=$(date -d "$end_time_formatted" +%s)

# 打印用于调试
echo "Start time formatted: $start_time_formatted"
echo "End time formatted: $end_time_formatted"
echo "Start seconds: $start_seconds"
echo "End seconds: $end_seconds"

# ...
# 循环遍历时间
for (( t=$start_seconds; t<=$end_seconds; t++ ))
do
    # 构造文件名
    file_time=$(date -d @$t +"%Y-%m-%d_%H-%M-%S")
    file_name="log_file_$file_time.log" # 根据您的文件名格式调整

    # 如果文件存在，则提取pid
    if [ -f "$file_name" ]; then
        pid=$(grep 'pid:' $file_name | awk '{print $2}')
        echo "File: $file_name, PID: $pid"
    fi
done

