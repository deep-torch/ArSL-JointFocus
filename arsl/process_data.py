import os
import argparse
from concurrent.futures import ThreadPoolExecutor


def process_session(session_path):
     for frame_idx, img_filename in enumerate(sorted(os.listdir(session_path))):
         if frame_idx % 10 != 0:
             img_path = os.path.join(session_path, img_filename)
             os.remove(img_path)

def main(args):
    sessions = []

    for signer in os.listdir(args.root_dir):
        # running code on colab sometimes create such folders
        if signer == '.ipynb_checkpoints':
            continue

        signer_path = os.path.join(args.root_dir, signer)

        for split in ['train', 'test']:
            split_path = os.path.join(signer_path, split)
            for sign in os.listdir(split_path):
                # running code on colab sometimes create such folders
                if sign == '.ipynb_checkpoints':
                    continue

                sign_path = os.path.join(split_path, sign)
                sessions.extend(
                    [os.path.join(sign_path, session)
                     for session in os.listdir(sign_path)]
                )

    with ThreadPoolExecutor() as executor:
        executor.map(process_session, sessions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download and extract dataset.')

    parser.add_argument('--root_dir', type=str, help='Data root directory')

    args = parser.parse_args()

    main(args)

    print(f'Successfully processed dataset')
