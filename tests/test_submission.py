# import pytest
import os
import re
import pathlib


def test_submission_dir():
    sub_folder = pathlib.Path(__file__).parent.parent.joinpath('submission')   
    for i, (root, dirs, files) in enumerate(os.walk(sub_folder)):
        # print(i)
        # print('root\n', root)
        # print('dirs\n', dirs)
        # print('files\n', files)
        if i == 0:
            assert 'YOUR-LAST-NAME' not in dirs, \
                "Change the YOUR-LAST NAME folder to your name"
            assert not files, "Files go below YOUR-LAST-NAME folder"
        if i == 1:
            assert any(
                re.match(r".zip", item) for item in dirs
            ), 'Did you submit a model .zip file?'
            assert 'SUBMISSION.md' in files, "SUBMISSION.md got moved?"
            assert any(
                re.search(r".mp4", item) for item in files
            ), "Please submit a .mp4"

# test_submission_dir()
