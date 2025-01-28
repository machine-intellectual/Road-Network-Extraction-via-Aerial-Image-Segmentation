import os
import requests


def download_file(url, save_dir):
    # Extract the file name from the URL
    #   https://maxar-opendata.s3.us-west-2.amazonaws.com/events/HurricaneHelene-Oct24/ard/16/120020230112/2023-04-18/10300100E5C9D100-visual.tif
    urls_segments = url.split("/")
    file_name = urls_segments[-1]
    date = urls_segments[-2]
    quad = urls_segments[-3]

    date_split = date.split("-")
    year, month, day = int(date_split[0]), int(date_split[1]), int(date_split[2])
    pre_post = "pre"


    if year == 2024 and (month == 9 and day >= 27 or month == 10):
        pre_post = "post"

    save_dir = f"{save_dir}/{pre_post}"

    save_path = f"{save_dir}/{quad}_{date}_{file_name}"

    if not os.path.exists(save_path):
        try:
            # Send GET request to the URL
            response = requests.get(url)

            # Check if the request was successful
            if response.status_code == 200:
                # Write the content to a file
                with open(save_path, 'wb') as f:
                    f.write(response.content)
            else:
                print(f"Failed to download {file_name}. Status code: {response.status_code}")
        except Exception as e:
            print(f"An error occurred while downloading {file_name}: {e}")


def download_files_from_urls(url_list, save_dir):
    # Make sure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(f"{save_dir}/pre")
        os.makedirs(f"{save_dir}/post")

    # Loop through each URL and download the file
    for url in url_list:
        download_file(url, save_dir)
        print('.', end='')


# Example usage
if __name__ == "__main__":

    download_directory = "datasets/maxtar/helene"

    with open("datasets/maxar_urls.txt", 'r') as file:
        urls = file.read().splitlines()

    download_files_from_urls(urls, download_directory)
