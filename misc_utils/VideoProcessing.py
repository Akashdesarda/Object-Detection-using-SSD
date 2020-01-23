import os
import imageio
class VideoProcessing:
    """Used for video processing. Video to frames and Frames to video are available.
       Parameters
        ----------
        video : str
            path to video
    """
    def __init__(self, video: str):
        
        self.video = video
        
    @staticmethod
    def existsFolder(path: str):
        if not os.path.exists(path):
            os.makedirs(path)
    
    @staticmethod
    def get_fps(video):
        fps = int(imageio.get_reader(video).get_meta_data()['fps'])
        return fps
        
    def generate_frames(self, export_path: str= "./assets/frames_data"):
        """Generate frames from given video
        
        Parameters
        ----------
        export_path : str, optional
            path to generate frames, by default "./assets/frames_data"
        """
        fps = VideoProcessing.get_fps(self.video)
        folder = self.video.split('.')[0]
        VideoProcessing.existsFolder(f"{export_path}/{folder}")
        print(f"[INFO]... Generating Frames from given video at {export_path}/{folder}")
        os.system(f"ffmpeg -i ./{self.video} -vf fps={fps} {export_path}{folder}/{folder}_%08d.jpg")
        
    def generate_video(self, import_path: str, export_path: str="./assets"):
        """Generate Video from frames
        
        Parameters
        ----------
        import_path : str
            path to directory where all frames are stored
        
        export_path : str, optional
            export video to directory, by default "./assets"
            
        Raises
        ------
        ValueError
            If given directory has no frames
        """
        fps = VideoProcessing.get_fps(self.video)
        folder = self.video.split('.')[0]
        if len(os.listdir(f"{import_path}/{folder}")) == 0:
            raise ValueError("[ERROR]...Given directory is empty")
        print("[INFO]...Generating Video from frames in {import_path}/{folder}")
        VideoProcessing.existsFolder(f"./{export_path}/{folder}_result")
        os.system(f"ffmpeg -framerate {fps} -i {import_path}/{folder}/{folder}_%08d.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p ./{export_path}/{folder}_result/output.mp4")