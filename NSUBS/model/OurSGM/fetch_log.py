from NSUBS.src.utils import exec_cmd

proj = 'OurSGM'
server = 'anonymous@serverxxx.cs.ucla.edu'
remote_path = '/home/anonymous/GraphMatching/model/{}/logs'.format(proj)
local_path = '/home/anonymous/Documents/GraphMatching/model/{}/logs'.format(proj)
end_markers = ['final_test_pairs.klepto', 'exception.txt']

local_subfolder = ''


log_folder = 'our_train_imsm-youtube_dense_32_nn_120_2022-02-09T10-54-54.767545'

def main():
    fp = log_folder.replace('|', '\|').replace(' ', '\ ').replace("'", "\\'")
    # scp_cmd = f'scp -r "{server}:{remote_path}/{fp}" {local_path}/{local_subfolder}'
    scp_cmd = f"rsync -avzh --exclude='*.pickle' --exclude='*.gexf' {server}:{remote_path}/{fp} {local_path}/{local_subfolder}"
    exec_cmd(scp_cmd)


if __name__ == '__main__':
    main()
