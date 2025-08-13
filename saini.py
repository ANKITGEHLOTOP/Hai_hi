import os
import re
import time
import mmap
import datetime
import aiohttp
import aiofiles
import asyncio
import logging
import requests
import tgcrypto
import subprocess
import concurrent.futures
from math import ceil
from utils import progress_bar
from pyrogram import Client, filters
from pyrogram.types import Message
from io import BytesIO
from pathlib import Path  
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
from base64 import b64decode

# Global connection pool for better performance
session = None
async def get_session():
    global session
    if session is None:
        connector = aiohttp.TCPConnector(
            limit=100,  # Increased connection limit
            limit_per_host=30,  # More connections per host
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        timeout = aiohttp.ClientTimeout(total=300, connect=60)
        session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
    return session

# Global connection pool for better performance
session = None
async def get_session():
    global session
    if session is None:
        connector = aiohttp.TCPConnector(
            limit=100,  # Increased connection limit
            limit_per_host=30,  # More connections per host
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        timeout = aiohttp.ClientTimeout(total=300, connect=60)
        session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
    return session

# Optimized download function with parallel processing
async def optimized_download(url, name, max_workers=8):
    """Optimized download with parallel chunking for faster downloads"""
    session = await get_session()
    
    try:
        async with session.head(url) as response:
            if response.status != 200:
                raise Exception(f"HTTP {response.status}")
            
            total_size = int(response.headers.get('content-length', 0))
            if total_size == 0:
                # Fallback to regular download if content-length not available
                return await download(url, name)
            
            # Calculate optimal chunk size (1MB chunks for better performance)
            chunk_size = 1024 * 1024  # 1MB
            chunks = []
            
            # Create chunks for parallel download
            for i in range(0, total_size, chunk_size):
                end = min(i + chunk_size - 1, total_size - 1)
                chunks.append((i, end))
            
            # Download chunks in parallel
            semaphore = asyncio.Semaphore(max_workers)
            
            async def download_chunk(chunk_info):
                start, end = chunk_info
                async with semaphore:
                    headers = {'Range': f'bytes={start}-{end}'}
                    async with session.get(url, headers=headers) as resp:
                        if resp.status in [200, 206]:
                            return start, await resp.read()
                        else:
                            raise Exception(f"Chunk download failed: {resp.status}")
            
            # Execute parallel downloads
            tasks = [download_chunk(chunk) for chunk in chunks]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine chunks in order
            with open(f'{name}.pdf', 'wb') as f:
                for result in results:
                    if isinstance(result, Exception):
                        raise result
                    start, data = result
                    f.seek(start)
                    f.write(data)
            
            return f'{name}.pdf'
            
    except Exception as e:
        logging.error(f"Optimized download failed: {e}")
        # Fallback to regular download
        return await download(url, name)

def duration(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)

def get_mps_and_keys(api_url):
    response = requests.get(api_url)
    response_json = response.json()
    mpd = response_json.get('MPD')
    keys = response_json.get('KEYS')
    return mpd, keys
   
def exec(cmd):
        process = subprocess.run(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        output = process.stdout.decode()
        print(output)
        return output

def pull_run(work, cmds):
    with concurrent.futures.ThreadPoolExecutor(max_workers=work) as executor:
        print("Waiting for tasks to complete")
        fut = executor.map(exec,cmds)

async def aio(url,name):
    k = f'{name}.pdf'
    session = await get_session()
    async with session.get(url) as resp:
        if resp.status == 200:
            f = await aiofiles.open(k, mode='wb')
            await f.write(await resp.read())
            await f.close()
    return k

async def download(url,name):
    ka = f'{name}.pdf'
    session = await get_session()
    async with session.get(url) as resp:
        if resp.status == 200:
            f = await aiofiles.open(ka, mode='wb')
            await f.write(await resp.read())
            await f.close()
    return ka

async def pdf_download(url, file_name, chunk_size=1024 * 10):
    if os.path.exists(file_name):
        os.remove(file_name)
    
    session = await get_session()
    async with session.get(url) as resp:
        if resp.status == 200:
            with open(file_name, 'wb') as fd:
                async for chunk in resp.content.iter_chunked(chunk_size):
                    if chunk:
                        fd.write(chunk)
    return file_name   

def parse_vid_info(info):
    info = info.strip()
    info = info.split("\n")
    new_info = []
    temp = []
    for i in info:
        i = str(i)
        if "[" not in i and '---' not in i:
            while "  " in i:
                i = i.replace("  ", " ")
            i.strip()
            i = i.split("|")[0].split(" ",2)
            try:
                if "RESOLUTION" not in i[2] and i[2] not in temp and "audio" not in i[2]:
                    temp.append(i[2])
                    new_info.append((i[0], i[2]))
            except:
                pass
    return new_info

def vid_info(info):
    info = info.strip()
    info = info.split("\n")
    new_info = dict()
    temp = []
    for i in info:
        i = str(i)
        if "[" not in i and '---' not in i:
            while "  " in i:
                i = i.replace("  ", " ")
            i.strip()
            i = i.split("|")[0].split(" ",3)
            try:
                if "RESOLUTION" not in i[2] and i[2] not in temp and "audio" not in i[2]:
                    temp.append(i[2])
                    new_info.update({f'{i[2]}':f'{i[0]}'})
            except:
                pass
    return new_info

async def decrypt_and_merge_video(mpd_url, keys_string, output_path, output_name, quality="720", progress_message=None):
    try:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Optimized yt-dlp command for faster downloads
        cmd1 = f'yt-dlp -f "bv[height<={quality}]+ba/b" -o "{output_path}/file.%(ext)s" --allow-unplayable-format --no-check-certificate --external-downloader aria2c --downloader-args "aria2c: -x 16 -j 32 -s 16 -k 1M" "{mpd_url}"'
        print(f"Running command: {cmd1}")
        
        if progress_message:
            from utils import download_progress_bar
            import time
            start_time = time.time()
            await progress_message.edit(f"<blockquote>📥 **Downloading Video & Audio...**\n📁 **File:** {output_name}\n⏳ **Status:** Starting download...</blockquote>")
        
        os.system(cmd1)
        
        avDir = list(output_path.iterdir())
        print(f"Downloaded files: {avDir}")
        print("Decrypting")
        
        if progress_message:
            await progress_message.edit(f"<blockquote>🔓 **Decrypting Files...**\n📁 **File:** {output_name}\n⏳ **Status:** Processing video & audio...</blockquote>")

        video_decrypted = False
        audio_decrypted = False

        for data in avDir:
            if data.suffix == ".mp4" and not video_decrypted:
                cmd2 = f'mp4decrypt {keys_string} --show-progress "{data}" "{output_path}/video.mp4"'
                print(f"Running command: {cmd2}")
                os.system(cmd2)
                if (output_path / "video.mp4").exists():
                    video_decrypted = True
                data.unlink()
            elif data.suffix == ".m4a" and not audio_decrypted:
                cmd3 = f'mp4decrypt {keys_string} --show-progress "{data}" "{output_path}/audio.m4a"'
                print(f"Running command: {cmd3}")
                os.system(cmd3)
                if (output_path / "audio.m4a").exists():
                    audio_decrypted = True
                data.unlink()

        if not video_decrypted or not audio_decrypted:
            raise FileNotFoundError("Decryption failed: video or audio file not found.")

        # Optimized ffmpeg command for faster merging
        cmd4 = f'ffmpeg -i "{output_path}/video.mp4" -i "{output_path}/audio.m4a" -c copy -movflags +faststart "{output_path}/{output_name}.mp4"'
        print(f"Running command: {cmd4}")
        
        if progress_message:
            await progress_message.edit(f"<blockquote>🔗 **Merging Video & Audio...**\n📁 **File:** {output_name}\n⏳ **Status:** Finalizing...</blockquote>")
        
        os.system(cmd4)
        if (output_path / "video.mp4").exists():
            (output_path / "video.mp4").unlink()
        if (output_path / "audio.m4a").exists():
            (output_path / "audio.m4a").unlink()
        
        filename = output_path / f"{output_name}.mp4"

        if not filename.exists():
            raise FileNotFoundError("Merged video file not found.")
        
        if progress_message:
            await progress_message.edit(f"<blockquote>✅ **Download Complete!**\n📁 **File:** {output_name}\n📦 **Status:** Ready for upload</blockquote>")
            await asyncio.sleep(2)
            await progress_message.delete()

        cmd5 = f'ffmpeg -i "{filename}" 2>&1 | grep "Duration"'
        duration_info = os.popen(cmd5).read()
        print(f"Duration info: {duration_info}")

        return str(filename)

    except Exception as e:
        print(f"Error during decryption and merging: {str(e)}")
        raise

async def run(cmd):
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)

    stdout, stderr = await proc.communicate()

    print(f'[{cmd!r} exited with {proc.returncode}]')
    if proc.returncode == 1:
        return False
    if stdout:
        return f'[stdout]\n{stdout.decode()}'
    if stderr:
        return f'[stderr]\n{stderr.decode()}'

def old_download(url, file_name, chunk_size = 1024 * 10 * 10):
    if os.path.exists(file_name):
        os.remove(file_name)
    r = requests.get(url, allow_redirects=True, stream=True)
    with open(file_name, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                fd.write(chunk)
    return file_name

def human_readable_size(size, decimal_places=2):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB']:
        if size < 1024.0 or unit == 'PB':
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f} {unit}"

def time_name():
    date = datetime.date.today()
    now = datetime.datetime.now()
    current_time = now.strftime("%H%M%S")
    return f"{date} {current_time}.mp4"

# Optimized download function with better progress tracking and parallel processing
async def download_with_progress(url, cmd, name, progress_message=None):
    """
    Optimized download with real-time progress tracking and parallel processing
    """
    # Enhanced yt-dlp command for faster downloads on Heroku Standard 2X
    download_cmd = f'{cmd} -R 25 --fragment-retries 25 --external-downloader aria2c --downloader-args "aria2c: -x 16 -j 32 -s 16 -k 1M --max-connection-per-server=16 --min-split-size=1M --split=16"'
    global failed_counter
    print(download_cmd)
    logging.info(download_cmd)
    
    from utils import download_progress_bar
    import time
    import asyncio
    
    start_time = time.time()
    last_size = 0
    
    # Start the download process with optimized settings
    process = subprocess.Popen(download_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Enhanced progress monitoring
    while process.poll() is None:
        downloaded_size = 0
        file_found = False
        
        # Check for downloaded file with multiple extensions
        for ext in ['', '.webm', '.mkv', '.mp4', '.mp4.webm']:
            test_name = name + ext if ext else name
            if os.path.isfile(test_name):
                downloaded_size = os.path.getsize(test_name)
                file_found = True
                break
        
        if not file_found:
            name_base = name.split(".")[0]
            for ext in ['.mkv', '.mp4', '.mp4.webm']:
                test_name = name_base + ext
                if os.path.isfile(test_name):
                    downloaded_size = os.path.getsize(test_name)
                    file_found = True
                    break
        
        # Update progress with better estimation
        if file_found and downloaded_size > last_size and progress_message:
            estimated_total = downloaded_size * 1.05  # More accurate estimate
            await download_progress_bar(downloaded_size, estimated_total, progress_message, start_time, name)
            last_size = downloaded_size
        
        await asyncio.sleep(0.5)  # More frequent updates for better UX
    
    process.wait()
    
    if "visionias" in cmd and process.returncode != 0 and failed_counter <= 10:
        failed_counter += 1
        await asyncio.sleep(5)
        return await download_with_progress(url, cmd, name, progress_message)
    
    failed_counter = 0
    
    if progress_message:
        await progress_message.edit(f"<blockquote>✅ **Download Complete!**\n📁 **File:** {name}\n📦 **Status:** Ready for upload</blockquote>")
        await asyncio.sleep(2)
        await progress_message.delete()
    
    # Return the downloaded file path
    try:
        if os.path.isfile(name):
            return name
        elif os.path.isfile(f"{name}.webm"):
            return f"{name}.webm"
        name = name.split(".")[0]
        if os.path.isfile(f"{name}.mkv"):
            return f"{name}.mkv"
        elif os.path.isfile(f"{name}.mp4"):
            return f"{name}.mp4"
        elif os.path.isfile(f"{name}.mp4.webm"):
            return f"{name}.mp4.webm"

        return name
    except FileNotFoundError as exc:
        return os.path.isfile.splitext[0] + "." + "mp4"

async def download_video(url, cmd, name):
    download_cmd = f'{cmd} -R 25 --fragment-retries 25 --external-downloader aria2c --downloader-args "aria2c: -x 16 -j 32 -s 16 -k 1M --max-connection-per-server=16 --min-split-size=1M --split=16"'
    global failed_counter
    print(download_cmd)
    logging.info(download_cmd)
    
    from utils import download_progress_bar
    import time
    
    start_time = time.time()
    
    try:
        k = subprocess.run(download_cmd, shell=True)
        
        if "visionias" in cmd and k.returncode != 0 and failed_counter <= 10:
            failed_counter += 1
            await asyncio.sleep(5)
            return await download_video(url, cmd, name)
        failed_counter = 0
        
        try:
            if os.path.isfile(name):
                return name
            elif os.path.isfile(f"{name}.webm"):
                return f"{name}.webm"
            name = name.split(".")[0]
            if os.path.isfile(f"{name}.mkv"):
                return f"{name}.mkv"
            elif os.path.isfile(f"{name}.mp4"):
                return f"{name}.mp4"
            elif os.path.isfile(f"{name}.mp4.webm"):
                return f"{name}.mp4.webm"

            return name
        except FileNotFoundError as exc:
            return os.path.isfile.splitext[0] + "." + "mp4"
            
    except Exception as e:
        logging.error(f"Download error: {e}")
        return None

async def send_doc(bot: Client, m: Message, cc, ka, cc1, prog, count, name, channel_id):
    reply = await bot.send_message(channel_id, f"Downloading pdf:\n<pre><code>{name}</code></pre>")
    time.sleep(1)
    start_time = time.time()
    
    file_size_mb = get_file_size_mb(ka)
    if file_size_mb > 1990:
        size_info = f"\n\n**📦 File Size: {file_size_mb:.1f} MB**\n**⚠️ Large file - may take time to download**"
        await bot.send_document(ka, caption=cc1 + size_info)
    else:
        await bot.send_document(ka, caption=cc1)
    
    count+=1
    await reply.delete (True)
    time.sleep(1)
    os.remove(ka)
    time.sleep(3) 

def decrypt_file(file_path, key):  
    if not os.path.exists(file_path): 
        return False  

    with open(file_path, "r+b") as f:  
        num_bytes = min(28, os.path.getsize(file_path))  
        with mmap.mmap(f.fileno(), length=num_bytes, access=mmap.ACCESS_WRITE) as mmapped_file:  
            for i in range(num_bytes):  
                mmapped_file[i] ^= ord(key[i]) if i < len(key) else i 
    return True  

async def download_and_decrypt_video(url, cmd, name, key, progress_message=None):  
    video_path = await download_with_progress(url, cmd, name, progress_message)  
    
    if video_path:  
        if progress_message:
            await progress_message.edit(f"<blockquote>🔓 **Decrypting File...**\n📁 **File:** {name}\n⏳ **Status:** Processing...</blockquote>")
        
        decrypted = decrypt_file(video_path, key)  
        if decrypted:  
            print(f"File {video_path} decrypted successfully.")  
            if progress_message:
                await progress_message.edit(f"<blockquote>✅ **Download & Decrypt Complete!**\n📁 **File:** {name}\n📦 **Status:** Ready for upload</blockquote>")
                await asyncio.sleep(2)
                await progress_message.delete()
            return video_path  
        else:  
            print(f"Failed to decrypt {video_path}.")  
            return None  

# Optimized video sending with better chunking and parallel processing
async def send_vid(bot: Client, m: Message, cc, filename, thumb, name, prog, channel_id):
    # Generate thumbnail with optimized ffmpeg command
    subprocess.run(f'ffmpeg -i "{filename}" -ss 00:00:10 -vframes 1 -q:v 2 "{filename}.jpg"', shell=True)
    await prog.delete (True)
    reply1 = await bot.send_message(channel_id, f"**📩 Uploading Video 📩:-**\n<blockquote>**{name}**</blockquote>")
    reply = await m.reply_text(f"**Generate Thumbnail:**\n<blockquote>**{name}**</blockquote>")
    
    try:
        if thumb == "/d":
            thumbnail = f"{filename}.jpg"
        else:
            thumbnail = thumb
            
    except Exception as e:
        await m.reply_text(str(e))
      
    dur = int(duration(filename))
    start_time = time.time()

    try:
        # Optimized video upload with better streaming support
        await bot.send_video(
            channel_id, 
            filename, 
            caption=cc, 
            supports_streaming=True, 
            height=720, 
            width=1280, 
            thumb=thumbnail, 
            duration=dur, 
            progress=progress_bar, 
            progress_args=(reply, start_time)
        )
    except Exception:
        # Fallback to document upload
        await bot.send_document(channel_id, filename, caption=cc, progress=progress_bar, progress_args=(reply, start_time))
    
    os.remove(filename)
    await reply.delete(True)
    await reply1.delete(True)
    os.remove(f"{filename}.jpg")

def get_file_size_mb(file_path):
    """Get file size in MB"""
    if os.path.exists(file_path):
        size_bytes = os.path.getsize(file_path)
        size_mb = size_bytes / (1024 * 1024)
        return size_mb
    return 0

# Optimized video splitting with better chunking algorithm
def split_video(input_file, output_prefix, max_size_mb=1990):
    """
    Optimized video splitting with better chunking for faster processing
    """
    try:
        # Get video duration using ffprobe
        probe_cmd = f'ffprobe -v quiet -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{input_file}"'
        duration = float(subprocess.check_output(probe_cmd, shell=True).decode().strip())
        
        file_size_mb = get_file_size_mb(input_file)
        num_parts = int(file_size_mb / max_size_mb) + 1
        duration_per_part = duration / num_parts
        
        output_files = []
        
        # Use parallel processing for faster splitting
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for i in range(num_parts):
                start_time = i * duration_per_part
                end_time = (i + 1) * duration_per_part if i < num_parts - 1 else duration
                output_file = f"{output_prefix}_part{i+1:02d}.mp4"
                
                # Optimized ffmpeg command for faster splitting
                split_cmd = f'ffmpeg -i "{input_file}" -ss {start_time} -to {end_time} -c copy -avoid_negative_ts make_zero "{output_file}" -y'
                
                future = executor.submit(subprocess.run, split_cmd, shell=True, check=True)
                futures.append((future, output_file))
            
            # Wait for all splits to complete
            for future, output_file in futures:
                future.result()
                if os.path.exists(output_file):
                    split_size = get_file_size_mb(output_file)
                    if split_size > max_size_mb:
                        # Re-encode with optimized settings if still too large
                        reencode_cmd = f'ffmpeg -i "{output_file}" -c:v libx264 -preset fast -crf 23 -c:a aac -b:a 128k "{output_file}_temp.mp4" -y'
                        subprocess.run(reencode_cmd, shell=True, check=True)
                        os.replace(f"{output_file}_temp.mp4", output_file)
                    
                    output_files.append(output_file)
        
        return output_files
        
    except Exception as e:
        logging.error(f"Error splitting video: {e}")
        return []

# Optimized video parts sending with parallel processing
async def send_video_parts(bot: Client, m: Message, cc, filename, thumb, name, prog, channel_id, max_size_mb=1990):
    """
    Optimized video parts sending with parallel processing for faster uploads
    """
    try:
        file_size_mb = get_file_size_mb(filename)
        
        if file_size_mb <= max_size_mb:
            await send_vid(bot, m, cc, filename, thumb, name, prog, channel_id)
            return
        
        split_msg = await bot.send_message(channel_id, f"**📦 File too large ({file_size_mb:.1f} MB), splitting into parts...**\n<blockquote>**{name}**</blockquote>")
        
        # Generate thumbnail
        subprocess.run(f'ffmpeg -i "{filename}" -ss 00:00:10 -vframes 1 -q:v 2 "{filename}.jpg"', shell=True)
        
        try:
            if thumb == "/d":
                thumbnail = f"{filename}.jpg"
            else:
                thumbnail = thumb
        except Exception as e:
            thumbnail = f"{filename}.jpg"
        
        # Split the video
        output_prefix = filename.rsplit('.', 1)[0]
        split_files = split_video(filename, output_prefix, max_size_mb)
        
        if not split_files:
            await split_msg.edit("**❌ Failed to split video**")
            return
        
        await split_msg.edit(f"**✅ Split into {len(split_files)} parts**\n<blockquote>**{name}**</blockquote>")
        await asyncio.sleep(2)
        await split_msg.delete()
        
        # Send parts with optimized parallel processing
        async def send_part(part_file, part_index, total_parts):
            if not os.path.exists(part_file):
                return
                
            part_size = get_file_size_mb(part_file)
            part_caption = f"{cc}\n\n**📦 Part {part_index}/{total_parts}**\n**"
            
            upload_msg = await bot.send_message(channel_id, f"**📤 Uploading Part {part_index}/{total_parts}**\n<blockquote>**{name}**</blockquote>")
            
            try:
                dur = int(duration(part_file))
                start_time = time.time()
                
                await bot.send_video(
                    channel_id, 
                    part_file, 
                    caption=part_caption, 
                    supports_streaming=True, 
                    height=720, 
                    width=1280, 
                    thumb=thumbnail, 
                    duration=dur, 
                    progress=progress_bar, 
                    progress_args=(upload_msg, start_time)
                )
            except Exception:
                await bot.send_document(
                    channel_id, 
                    part_file, 
                    caption=part_caption, 
                    progress=progress_bar, 
                    progress_args=(upload_msg, start_time)
                )
            
            await upload_msg.delete(True)
            os.remove(part_file)
            await asyncio.sleep(0.5)  # Reduced delay for faster processing
        
        # Send parts with limited concurrency to avoid rate limits
        semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent uploads
        
        async def send_part_with_semaphore(part_file, part_index, total_parts):
            async with semaphore:
                await send_part(part_file, part_index, total_parts)
        
        # Create tasks for parallel sending
        tasks = [
            send_part_with_semaphore(part_file, i+1, len(split_files))
            for i, part_file in enumerate(split_files)
        ]
        
        # Execute parallel uploads
        await asyncio.gather(*tasks)
        
        # Clean up
        os.remove(filename)
        if os.path.exists(f"{filename}.jpg"):
            os.remove(f"{filename}.jpg")
            
    except Exception as e:
        logging.error(f"Error in send_video_parts: {e}")
        # Fallback to original method
        await send_vid(bot, m, cc, filename, thumb, name, prog, channel_id)

# Cleanup function for session
async def cleanup_session():
    global session
    if session:
        await session.close()
        session = None
