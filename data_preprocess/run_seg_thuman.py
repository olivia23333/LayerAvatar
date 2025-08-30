import os
import subprocess
import multiprocessing

def run_on_gpu(base_dir, folders, gpu_id):
    for folder in folders:
        # Modify the command to specify the GPU
        folder_path = os.path.join(base_dir, folder)
        command = f"CUDA_VISIBLE_DEVICES={gpu_id} {program_command} --input {folder_path}/18views_3_wbg/image --batch-size=4 --output-root {folder_path}/18views_3_wbg/mask"
        print(f"Running: {command}")
        subprocess.run(command, shell=True)

if __name__ == "__main__":
    # Define the base directory containing your folders
    base_dir = '/xxx/HumanData/THuman/THuman2.0_Release'
    # Define the command to run your program
    program_command = 'python demo/vis_seg.py /xxx/sapiens/checkpoint/seg/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2'  # Replace with your actual command

    # Get the list of folders
    folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

    # Split the folders into 2 parts
    num_gpus = 2
    split_folders = [folders[i::num_gpus] for i in range(num_gpus)]

    processes = []
    for gpu_id, folder_batch in enumerate(split_folders):
        p = multiprocessing.Process(target=run_on_gpu, args=(base_dir, folder_batch, gpu_id))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()