from pytube import YouTube

# Replace 'video_url' with the URL of the YouTube video
video_url = 'https://www.youtube.com/shorts/zmF9wt0F0XQ'
yt = YouTube(video_url)
video = yt.streams.filter(file_extension='mp4', res='720p').first()
video.download('path_to_download_directory')