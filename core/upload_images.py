import os
import time
import glob
import firebase_admin
from firebase_admin import credentials, storage
from google.cloud import storage as st
import requests

images_url_lst = []


def upload_images(image_path, user_id, keys_path=None):
    if not firebase_admin._apps:
        cred = credentials.Certificate(keys_path)
        firebase_admin.initialize_app(cred, {'storageBucket': 'lookat-5d200.appspot.com'})
        bucket = storage.bucket()
        blobs = bucket.list_blobs(prefix=f'users_images/{user_id}/users_predicted_images/')
        for blob in blobs:
            blob.delete()
        print("All files in the directory have been removed!")
        time.sleep(1)
        files_names = os.listdir(image_path)
        for file in files_names:
            print(file)
            file_path = os.path.join(image_path, file)
            blob = bucket.blob(f'users_images/{user_id}/users_predicted_images/{file}')
            blob.upload_from_filename(file_path)
            images_url = blob.public_url
            images_url_lst.append(images_url)
            print(f'Uploaded successfully....\n{images_url_lst}')
            return images_url_lst


if __name__ == '__main__':
    keys_path = './keys/serviceAccount.json'
    user_id = 'hMkxhu0iLsbfkl2aHFgc4RzQySu1'
    image_path = './images/request_images/ready image'
    upload_images(image_path=image_path, user_id=user_id, keys_path=keys_path)
    print(data)
