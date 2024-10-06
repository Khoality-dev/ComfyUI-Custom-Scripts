import os
import argparse
import json
from PIL import PngImagePlugin, Image
from PIL.ExifTags import TAGS
import re

def get_png_metadata(image_path) -> list:
    try:
        image = Image.open(image_path)
    except:
        print(f'Error opening {image_path}')
        return None
    metadata = image.info  # Extract PNG metadata
    if 'prompt' in metadata:
        
        json_string = json.loads(metadata['prompt'])
        prompts = []
        for key, value in json_string.items():
            try:
                prompt = value['inputs']['text']  
                prompts.append(prompt)
            except:
                pass
        prompts = [item for item in prompts if item]
        return prompts
    elif 'parameters' in metadata:
        json_string = metadata['parameters']
        prompts = json_string.split('Steps: ')[0].split('Negative prompt: ')
        prompts = [item for item in prompts if item]
        return prompts

def get_jpeg_metadata(image_path) -> list:
    try:
        image = Image.open(image_path)
    except:
        print(f'Error opening {image_path}')
        return None
    try:
        exif_data = image._getexif()
    except:
        print(f'Error opening {image_path}')
        return None
    
    if exif_data:
        # Loop through the EXIF data and extract tags
        for tag_id, value in exif_data.items():
            # Get the tag name
            tag_name = TAGS.get(tag_id, tag_id)

            # Check if this tag is a comment
            if tag_name == 'UserComment':
                if value.startswith(b'UNICODE'):
                    value = value[7:]
                try:
                    comment = value.decode('utf-16', 'ignore')[1:]
                except AttributeError:
                    return None
                prompts = comment.split('Steps: ')[0].split('Negative prompt: ')
                prompts = [item for item in prompts if item]
                return prompts
    else:
        print(f'Error opening {image_path}')
        return None


def list_files(data_dir):
    for root, subdirs, files in os.walk(data_dir, followlinks=True):
        for file in files:
            if os.path.isfile(os.path.join(root, file)):
                yield os.path.join(root, file)

def main(args):
    data_dir = args.data_dir
    files = list_files(data_dir)
    dataset = []
    for file in files:
        prompts = get_png_metadata(file)
        if prompts is None:
            prompts = get_jpeg_metadata(file)

        if prompts is not None:
            prompts = [re.sub(r'<[^>]+>', '', prompt) for prompt in prompts]
            prompts = [re.sub(r',\s*,', ',', cleaned_string) for cleaned_string in prompts]
            prompts = [re.sub(r'^\s*,|,\s*$', '', cleaned_string) for cleaned_string in prompts]
            dataset += prompts
    
    dataset = list(set(dataset))

    with open('prompts.json', 'w') as f:
        json.dump({'data': dataset}, f, indent=4)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing the data')
    args = parser.parse_args()
    main(args)