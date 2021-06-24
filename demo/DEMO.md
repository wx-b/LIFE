# Video demo

We introduce how to play a movie on the paper as the presented video demo in this tutorial.

### Step 1: Prepare data
1. An image or marker for dense tracking. We provide [The Starry Night](assets/imgs/TheStarryNight.png) as an example.
2. Print the image onto a paper and record a video for it.
3. Select a movie clip.

### Step 2: Data process
We can extract frames from the video and movie via ffmpeg:
```
ffmpeg -i video.mp4 -r 30/1 video/%06d.png
```

We provide a processed data package [here](https://drive.google.com/drive/folders/1cDrl6P291UbN-jyDd3s-GB-ddyP6NeWZ?usp=sharing).

### Step 3: Warp images by predicted flows
Run the script with proper parameters:
```
python demo.py
```

### Step 4: Synthesize the final video 
Merge generated images into a video.
```
ffmpeg -r 30/1 -i data/output/%06d.png -c:v libx264 -vf fps=30 -pix_fmt yuv420p demo.mp4
```
