'''
Functions to support Google Drive in Colab
'''

def fileid(url):
    "Extract the file id from a Google Drive sharelink"
    x = url.split("/")
    return x[5]

def downloadlink(url):
    "Compute a link to the actual file content from the Google Drive sharelink"
    id = fileid(url)
    return "https://drive.google.com/uc?export=download&id="+id
