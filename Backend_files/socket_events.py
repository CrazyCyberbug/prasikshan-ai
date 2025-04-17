# this is the room aware version. 

import whisper
from flask import request
from flask_socketio import join_room, leave_room, send, emit, close_room
import io
import wave
import os
import uuid
import torchaudio
from datetime import datetime, timedelta
from database import room_sessions_collection, chat_history_collection, users_collection
from utils.model import pdf_translation
from utils.test import text_to_text_translation
# from final_server2 import transcribe_speech, transcribe_text_only
import torch
# english transcription = ""
import base64
import time
from utils.text_to_text import NLLB, seamless_nllb_lang_mapping
from utils.text_to_text import languages  as nllb_languages
# --------------------------UNCOMMENT LINE 378 to 395---------------------------------
import weasyprint
from flask_socketio import SocketIO
from pprint import pprint
from io import BytesIO
import torch
import numpy as np
import threading
import mimetypes


# pip install fpdf
# ============================================
import os
import soundfile as sf
from datetime import datetime
# ===========================================

# ----------------------------------------------------------
AUDIO_DIR = "audio_files"
if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR)


import requests
import json
import logging

# Configure logging
# logging.basicConfig(
#     filename='app.log',   # Log file name
#     level=logging.INFO,    # Logging level
#     format='%(asctime)s - %(levelname)s - %(message)s'  # Format of log messages
# )

# Writing logs
from utils.text_to_text import NLLB,seamless_nllb_lang_mapping
from utils.text_to_text import languages as nllb_languages

nllb = NLLB()

instructor_language = ""
is_recording = True

rooms = {}
frames = []
all_transcriptions = []
user_sockets = {}
user_session_details = {}
old_len = 0
room_chats = {} 
chat_history = {}
audio_buffer = np.array([], np.float32)


sampling_rate = 16000
batch_size = sampling_rate * 3
keep_samples = int(sampling_rate * 0.15)

now = None
languages = {
    "vietnamese": "vie",
    "kazakh": "kaz",
    "russian": "rus",
    "french": "fra",
    "bengali": "ben",
    "arabic": "arb",
    "english": "eng",
    "hindi": "hin",
    "telugu": "tel",
}



def resolve_duration(audio: list):
    return len(audio) / 16000

import torch
class Segment:
    def __init__(self, audio:list, speaker_id : str , room_id,  start= None, end = None):
        self.audio = audio 
        self.speaker = speaker_id
        self.speaker_username = ""
        self.speaker_language = ""
        self.room_id = room_id
        self.start = start
        self.end = end
        self.speaker_role = ""
        self.instructor_lang = instructor_language
        
        self.get_user_details()
                
    def get_user_details(self):
        user_details = user_session_details.get(self.speaker, None)
        if user_details:
            self.speaker_role = user_details.get("role", "Not Found")
            self.speaker_username = user_details.get('username', "Not Found")
            self.room_id = user_details.get('room_id', "Not Found")
            self.speaker_language = user_details.get('language', "Not Found")
            
    def to_dict(self):
        return {"speaker_id":self.speaker,
                "speaker_role": self.speaker_role,
                "speaker_username": self.speaker_username,
                "speaker_language": self.speaker_language,
                "instructor_language": instructor_language,
                "room_id": self.room_id,
                "audio":list(self.audio),
                "start": self.start,
                "end":self.end}
                
class Segments_handler:
    def __init__(self):
        self.segments = {}
        
    def add_segment(self, room_id, audio, speaker):
        audio = audio.tolist()
        print(f"type of audio now is {type(audio)}")
        duration = resolve_duration(audio = audio)
        print(f"got a chunk of duration {duration}")        
        if self.segments.get(room_id, None) is None:
            self.segments[room_id] = []
            self.segments[room_id].append(Segment(audio, speaker, 0, duration))
        else:            
            if self.segments.get(room_id, None)[-1].speaker == speaker:
                print("Same speaker", speaker)
                if duration is not None and self.segments.get(room_id, None)[-1].end is not None:
                    self.segments.get(room_id, None)[-1].end += duration
                
                self.segments.get(room_id, None)[-1].audio.extend(audio)
                print("-----------------extended an existing segmentz ---------------")
            
            else:
                print("\n" * 5)
                print("speaker  has been changed")
                print("\n" * 5)
                if duration is not None and self.segments.get(room_id, None)[-1].end is not None:                    
                    prev_end = self.segments.get(room_id, None)[-1].end              
                    start, end  = prev_end, prev_end + duration
                else:
                    start, end = None, None
                    
                self.segments.get(room_id, None).append(Segment(audio, speaker, start, end))
                print("-----------------added a new segment---------------")

    def get_segments(self, room_id, clear = True):
        data = [segment.to_dict() for segment in self.segments.get(room_id, None)]
        if clear:
            self.segments = {}
        return data
        
    def transcribe_segments(self, room_id):   
        # load the data.
        data = self.get_segments(room_id = room_id)
        model = whisper.load_model("large", download_root="whipser", device = torch.device('cuda'))
        for segment in data:
            print("\n")
            # extracting info from segments
            audio = np.array(segment['audio'], dtype = np.float32)
            speaker_id = segment['speaker_id']
            speaker_role = segment['speaker_role']
            speaker_username = segment['speaker_username']
            speaker_language = segment['speaker_language']
            instructor_language = segment['instructor_language']
            
            
            # resample audio.
            tensor_samples = torch.tensor(audio, dtype=torch.float32)
            audio = torchaudio.functional.resample(tensor_samples, orig_freq=48000, new_freq=16_000).cpu().numpy()    
            
            # transcribe audio.
            translations = model.transcribe(audio, task = "translate")
            
            # formatting output text for readability. 
            # text = '.\n'.join(translations['text'].split("."))
            # print(f"[{segment['speaker_role']}]: {text}")
            
            sub_segments = [sgmnt['text'] for sgmnt in translations['segments']]
            
            
            
            # update the db
            for text in sub_segments:
                print(f"\n [{speaker_username}]{text}")
                
                room_exists = room_sessions_collection.find_one({"room": room_id})               
                
                
                transcription_data = {
                        "translated_text": text,
                        "speaker_id" :  speaker_id, 
                        "speaker_username" : speaker_username,
                        "speaker_role" : speaker_role,
                        "speaker_language_selection": languages.get(speaker_language, speaker_language),
                        "timestamp": datetime.now().isoformat()}                
                                    
                # action : send --------------if instrutctor speaks
                # action : received ---------- if any student speaks
                # instructor_language = "English"
                
                
                instructor_lang = languages.get(instructor_language, "eng")
                      
                
                print(instructor_lang,"*" * 50)
                
                
                
                
                # room_sessions_collection.update_one(
                #     {"room": room_id, "user_id": speaker_id,"instructor_lang": languages.get(instructor_lang, "NA")},
                #     {"$push": {"transcriptions": transcription_data}},
                #     upsert=True
                # )
                if room_exists:
                    # If room exists, update the transcriptions array
                    room_sessions_collection.update_one(
                        {"room": room_id},
                        {"$push": {"transcriptions": transcription_data}}
                    )
                else:
                    # If room does not exist, create a new document
                    new_document = {
                        "room": room_id,
                        "user_id": speaker_id,
                        "instructor_lang": instructor_lang,
                        "transcriptions": [transcription_data]  # Start with the first transcription
                    }
                    room_sessions_collection.insert_one(new_document)
     
        model = model.to("cpu")       
        del model      
        torch.cuda.empty_cache()
        print("...trancription thread shutting down...")
                
    def dump_segments(self, room_id = None, clear = False):
        import json
        print("dump segments is called")        
        if room_id is None:
            room_id = "<----test_room_id---->" 
        
        if self.segments.get(room_id, None) is None:
            return

        dump_folder = "dump_json"
        os.makedirs(dump_folder, exist_ok=True) 
        file_path = os.path.join(dump_folder, f"{room_id}_segments.json")
      
        data = [segment.to_dict() for segment in self.segments.get(room_id, None)]
        if clear:
            self.segments = {}
        # print(data)
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)
            absolute_path = os.path.abspath(f"{room_id}_segments.json")
            print(f"segments data has been dumped into {absolute_path}")
            
        transcription_thread = threading.Thread(target=self.transcribe_segments, args=(room_id,))
        transcription_thread.start()
  
segment_handler = Segments_handler()

class AudioConfig:
    SAMPLERATE = 16000
    CHANNELS = 1

class ServerConfig:
    server_host = "127.0.0.1"
    server_port = "8090"
    server_url = f"http://{server_host}:{server_port}/speech-translation"
    ws_server_url = "ws://localhost:8004/ws"
    headers = {"Content-Type": "application/json"}

def get_unique_words_percentage(sent):
    # lower
    sent = sent.lower()
    
    # remove punctuation
    sent = sent.replace(".", "")
    sent = sent.replace(",", "")

    words = sent.split()
    unique_words = list(set(words))
    unique_percent = len(unique_words)/max(len(words), 1) 
    return unique_percent


import asyncio
import websockets
import traceback

from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

vad_model = load_silero_vad()

# # This is the iteration that works works with ws,
# and restarts properly after the mic is turnedoff once.
# class AudioHandler:
#     def __init__(self):
#         # Room-specific data structures
#         self.threads = {}  # {room_id: {lang_code: thread}}
#         self.audio_buffers = {}  # {room_id: {lang_code: buffer}}
#         self.buffer_locks = {}  # {room_id: {lang_code: lock}}
#         self.language_mapping = {}  # {room_id: {lang_code: [user_ids]}}
        
#         self.stop_thread_flags = {}  # {room_id: bool}
#         self.processing_complete = {}  # {room_id: {lang_code: Event}}
#         self.batch_size = AudioConfig.SAMPLERATE * 1
#         self.socketio = None
        
#         self.current_speaker_id = {}
#         self.user_counts = {}  # {room_id: count}
#         self.shutdown_in_progress = {}  # {room_id: bool}
        
#         # WebSocket connections
#         self.ws_connections = {}  # {room_id: {lang_code: websocket}}
#         self.ws_locks = {}  # {room_id: {lang_code: lock}}
        
#         # Event loop for asyncio
#         self.loop = asyncio.new_event_loop()
#         self.loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
#         self.loop_thread.start()
    
#     def _run_event_loop(self):
#         """Run the asyncio event loop in a separate thread."""
#         asyncio.set_event_loop(self.loop)
#         self.loop.run_forever()

#     def handle_audio(self, data, socketio, room_id):
#         """Handle incoming audio data for a specific room."""
#         if not self.socketio:
#             self.socketio = socketio

#         # Don't accept new audio if shutdown is in progress
#         if self.shutdown_in_progress.get(room_id, False):
#             return

#         # Start threads if not already running for this room
#         if room_id not in self.threads:
#             print(f"Starting new threads for room {room_id}")
#             self.start_threads(room_id, start_up=True)
        
#         if room_id in self.current_speaker_id:
#             segment_handler.add_segment(room_id=room_id,
#                                         audio=data,
#                                         speaker=self.current_speaker_id[room_id])

#             # Add audio data to buffers for each language in the room
#             for lang_code in self.language_mapping.get(room_id, {}):
#                 self.add_audio_to_buffer(room_id, lang_code, data)
#         else:
#             print(f"Warning: No current speaker ID set for room {room_id}")

#     async def establish_websocket_connection(self, room_id, lang_code):
#         """Establish WebSocket connection for a specific room and language."""
#         try:
#             print(f"Attempting to establish WebSocket connection for room {room_id}, language {lang_code}")
#             ws = await websockets.connect(ServerConfig.ws_server_url, max_size=1024*1024*10)
            
#             # Initialize WebSocket connection
#             room_user_id = self.get_room_user_id(room_id, lang_code)
            
#             # Send initial message to reset buffers
#             payload = {
#                 "client_id": room_user_id,
#                 "audio_data": [],
#                 "sampling_rate": AudioConfig.SAMPLERATE,
#                 "tgt_lang": lang_code,
#                 "reset_buffers": "true"
#             }
#             await ws.send(json.dumps(payload))
#             print(f"Initial reset message sent for {room_user_id}")
            
#             # Wait for initial response
#             response = await ws.recv()
#             print(f"WebSocket connection established for room {room_id}, language {lang_code}")
            
#             return ws
#         except Exception as e:
#             print(f"Error establishing WebSocket connection for room {room_id}, language {lang_code}: {e}")
#             return None

#     async def send_audio_via_websocket(self, ws, room_user_id, audio_samples, tgt_lang, reset_buffers="false"):
#         """Send audio data via WebSocket and receive response."""
#         if ws is None:
#             print(f"Cannot send audio: WebSocket connection is None for {room_user_id}")
#             return None
            
#         try:
#             payload = {
#                 "client_id": room_user_id,
#                 "audio_data": audio_samples.tolist(),
#                 "sampling_rate": AudioConfig.SAMPLERATE,
#                 "tgt_lang": tgt_lang,
#                 "reset_buffers": reset_buffers
#             }
            
#             await ws.send(json.dumps(payload))
#             response = await ws.recv()
#             return json.loads(response)
#         except Exception as e:
#             print(f"Error sending audio via WebSocket for {room_user_id}: {e}")
#             return None

#     async def process_inference_async(self, room_id, lang_code, tgt_lang):
#         """Async version of process_inference using WebSockets."""
#         global user_sockets
        
#         print(f"Starting WebSocket inference thread for room {room_id}, language {lang_code}")
        
#         # Check if we should stop before even starting
#         if self.stop_thread_flags.get(room_id, False):
#             print(f"Thread for {room_id}/{lang_code} asked to stop before starting")
#             return
        
#         try:
#             # Establish WebSocket connection
#             ws = await self.establish_websocket_connection(room_id, lang_code)
#             if not ws:
#                 print(f"Failed to establish WebSocket connection for room {room_id}, language {lang_code}")
#                 return
            
#             # Store the websocket connection - use locks to prevent race conditions
#             if room_id in self.ws_locks and lang_code in self.ws_locks[room_id]:
#                 with self.ws_locks[room_id][lang_code]:
#                     if room_id in self.ws_connections and lang_code in self.ws_connections[room_id]:
#                         self.ws_connections[room_id][lang_code] = ws
            
#             while not self.stop_thread_flags.get(room_id, False):
#                 # Check if the connection is still valid
#                 if ws.close_code is not None:
#                     print(f"WebSocket closed unexpectedly for {room_id}/{lang_code}, attempting to reconnect")
#                     ws = await self.establish_websocket_connection(room_id, lang_code)
#                     if not ws:
#                         print(f"Failed to re-establish WebSocket connection for {room_id}/{lang_code}")
#                         break
                    
#                     with self.ws_locks[room_id][lang_code]:
#                         self.ws_connections[room_id][lang_code] = ws
                
#                 samples = np.array([], np.float32)
                
#                 # Process audio samples if available
#                 if room_id in self.buffer_locks and lang_code in self.buffer_locks[room_id]:
#                     with self.buffer_locks[room_id][lang_code]:
#                         if room_id in self.audio_buffers and lang_code in self.audio_buffers[room_id]:
#                             buffer = self.audio_buffers[room_id][lang_code]
#                             if len(buffer) > 0:
#                                 # Process in smaller chunks to ensure smooth handling
#                                 chunk_size = min(self.batch_size, len(buffer))
#                                 samples = buffer[:chunk_size]
#                                 self.audio_buffers[room_id][lang_code] = buffer[chunk_size:]
                
#                 if len(samples) > 0:
#                     recipient_ids = []
#                     if room_id in self.language_mapping and lang_code in self.language_mapping[room_id]:
#                         recipient_ids = self.language_mapping[room_id].get(lang_code, [])
                    
#                     room_user_id = self.get_room_user_id(room_id, lang_code)
                    
#                     # Process the audio chunk using WebSocket
#                     if room_id in self.ws_locks and lang_code in self.ws_locks[room_id]:
#                         with self.ws_locks[room_id][lang_code]:
#                             if room_id in self.ws_connections and lang_code in self.ws_connections[room_id]:
#                                 response = await self.send_audio_via_websocket(
#                                     self.ws_connections[room_id][lang_code],
#                                     room_user_id,
#                                     samples,
#                                     tgt_lang
#                                 )
                                
#                                 if response:
#                                     self.handle_inference_response_ws(response, room_id, lang_code, recipient_ids)
                    
 
            
#             # Clean up WebSocket connection
#             print(f"Closing WebSocket for {room_id}/{lang_code}")
#             if ws.close_code is not None:
#                 await ws.close()
            
#             # Signal completion before exiting
#             if room_id in self.processing_complete and lang_code in self.processing_complete[room_id]:
#                 print(f"Setting completion event for room {room_id}, language {lang_code}")
#                 self.processing_complete[room_id][lang_code].set()
                
#         except Exception as e:
#             print(f"Exception in process_inference_async for room {room_id}, language {lang_code}: {e}")
#             traceback.print_exc()
#             if room_id in self.processing_complete and lang_code in self.processing_complete[room_id]:
#                 self.processing_complete[room_id][lang_code].set()

#     def process_inference_wrapper(self, room_id, lang_code, tgt_lang):
#         """Wrapper to run async process_inference in a sync context."""
#         future = asyncio.run_coroutine_threadsafe(
#             self.process_inference_async(room_id, lang_code, tgt_lang),
#             self.loop
#         )
        
#         # Handle any exceptions from the future
#         try:
#             future.result()
#         except Exception as e:
#             print(f"Error in process_inference_wrapper: {e}")
#             traceback.print_exc()

#     async def stop_websocket_connections(self, room_id):
#         """Close all WebSocket connections for a room."""
#         if room_id in self.ws_connections:
#             for lang_code, ws in list(self.ws_connections[room_id].items()):
#                 try:
#                     if ws and  ws.close_code  is not None:
#                         # Send final cleanup message
#                         room_user_id = self.get_room_user_id(room_id, lang_code)
#                         await self.send_audio_via_websocket(
#                             ws, 
#                             room_user_id, 
#                             np.array([], np.float32), 
#                             "eng", 
#                             reset_buffers="true"
#                         )
#                         await ws.close()
#                 except Exception as e:
#                     print(f"Error closing WebSocket for {room_id}, {lang_code}: {e}")

#     def stop_threads(self, room_id=None):
#         """Stop threads gracefully, ensuring all buffered audio is processed."""
#         if room_id:
#             print(f"Initiating shutdown for room {room_id}")
            
#             # First check if the room exists in our data structures
#             if room_id not in self.threads:
#                 print(f"No threads found for room {room_id}, skipping shutdown")
#                 return
                
#             # Log audio buffer status
#             if room_id in self.audio_buffers:
#                 for lang_code, buffer in self.audio_buffers[room_id].items():
#                     print(f"{lang_code} has {len(buffer)/16000} seconds of unprocessed audio during shut down")
            
#             # Mark shutdown in progress to prevent new audio from being added
#             self.shutdown_in_progress[room_id] = True
            
#             # Set stop flag for the room to signal threads to terminate
#             self.stop_thread_flags[room_id] = True          
            
#             # Initialize completion events for each language
#             self.processing_complete[room_id] = {
#                 lang_code: threading.Event()
#                 for lang_code in self.threads.get(room_id, {}).keys()
#             }
            
#             # Wait for all threads to finish processing remaining audio
#             print(f"Waiting for processing completion in room {room_id}")
#             for lang_code, event in self.processing_complete[room_id].items():
#                 # Longer timeout to ensure processing completes
#                 if not event.wait(timeout=10.0):
#                     print(f"Warning: Timeout waiting for {lang_code} processing in room {room_id}")
#             print(f"All processes have completed. Initializing shutdown.")  
            
#             # Close all WebSocket connections
#             future = asyncio.run_coroutine_threadsafe(
#                 self.stop_websocket_connections(room_id),
#                 self.loop
#             )
            
#             # Wait for the future to complete with a timeout
#             try:
#                 future.result(timeout=5.0)
#             except Exception as e:
#                 print(f"Error stopping WebSocket connections: {e}")
            
#             # Check remaining audio after processing
#             if room_id in self.audio_buffers:
#                 for lang_code, buffer in self.audio_buffers[room_id].items():
#                     print(f"{lang_code} has {len(buffer)/16000} seconds of unprocessed audio after shut down")
            
#             # Clean up room resources
#             print(f"Cleaning up resources for room {room_id}")
            
#             # Clean up room-specific data structures
#             if room_id in self.threads:
#                 # Stop all threads first
#                 for thread in self.threads[room_id].values():
#                     if thread.is_alive():
#                         thread.join(timeout=2.0)
                
#                 # Now clean up the data structures
#                 self.threads.pop(room_id, None)
#                 self.audio_buffers.pop(room_id, None)
#                 self.buffer_locks.pop(room_id, None)
#                 self.language_mapping.pop(room_id, None)
#                 self.processing_complete.pop(room_id, None)
#                 self.shutdown_in_progress.pop(room_id, None)
#                 self.ws_connections.pop(room_id, None)
#                 self.ws_locks.pop(room_id, None)
#                 self.stop_thread_flags.pop(room_id, None)
#                 self.current_speaker_id.pop(room_id, None)
                
#             print(f"Shutdown complete for room {room_id}")
#         else:
#             # Stop all threads across all rooms
#             for room_id in list(self.threads.keys()):
#                 self.stop_threads(room_id)

#     # Helper methods
#     def get_room_user_id(self, room_id, user_id):
#         return f"{room_id}_{user_id}"

#     def has_remaining_audio(self, room_id, lang_code):
#         """Check if there is remaining audio for processing."""
#         if room_id in self.buffer_locks and lang_code in self.buffer_locks[room_id]:
#             with self.buffer_locks[room_id][lang_code]:
#                 if room_id in self.audio_buffers and lang_code in self.audio_buffers[room_id]:
#                     return len(self.audio_buffers[room_id][lang_code]) > 0
#         return False

#     def add_audio_to_buffer(self, room_id, lang_code, audio):
#         """Add audio data to the buffer for processing."""
#         if room_id not in self.buffer_locks:
#             self.buffer_locks[room_id] = {}
#             self.audio_buffers[room_id] = {}
            
#         if lang_code not in self.buffer_locks[room_id]:
#             self.buffer_locks[room_id][lang_code] = threading.Lock()
#             self.audio_buffers[room_id][lang_code] = np.array([], np.float32)
            
#         tensor_samples = torch.tensor(audio, dtype=torch.float32)
#         audio = torchaudio.functional.resample(tensor_samples, orig_freq=48000, new_freq=16_000).cpu().numpy()
        
#         with self.buffer_locks[room_id][lang_code]:
#             if room_id in self.language_mapping and lang_code in self.language_mapping[room_id]:
#                 if len(self.language_mapping[room_id][lang_code]) > 0:
#                     self.audio_buffers[room_id][lang_code] = np.append(self.audio_buffers[room_id][lang_code], audio)

#     def handle_inference_response_ws(self, response, room_id, lang_code, recipient_ids):
#         """Handle inference response from WebSocket."""
#         try:
#             transcriptions = response.get('transcriptions', "")
#             if transcriptions:
#                 text, audio = eval(transcriptions)
#                 print(text)
#                 if text[0] != "[" and get_unique_words_percentage(text) > 0.4:
#                     self.broadcast_results(room_id, lang_code, text, audio, recipient_ids)
#         except Exception as e:
#             print(f"Error handling WebSocket inference response: {e}")

#     def broadcast_results(self, room_id, lang_code, text, audio, recipient_ids):
#         """Broadcast translation results to recipients."""
#         for recipient_id in recipient_ids:
#             target_socket_id = user_sockets.get(recipient_id)
#             try:
#                 # Only send audio to recipients who aren't the speaker
#                 if recipient_id != self.current_speaker_id.get(room_id, "unknown"):
#                     # Broadcast audio
#                     audio_np = np.array(audio, dtype=np.float32)
#                     audio_file = f"{room_id}_{lang_code}_translated_audio.wav"
#                     sf.write(audio_file, audio_np, 16000)
                    
#                     with open(audio_file, "rb") as f:
#                         self.socketio.emit(
#                             "receive_audio",
#                             {"audio": f.read()},
#                             room=target_socket_id
#                         )
                
#                 # Broadcast text to all recipients
#                 print(f"inside broadcast {text}")
#                 self.socketio.emit(
#                     "transcription",
#                     {
#                         "english": "",
#                         "translated": text,
#                         "sender_user_id": self.current_speaker_id.get(room_id)
#                     },
#                     to=target_socket_id,
#                 )
#             except Exception as e:
#                 print(f"Broadcast error: {e}")

#     def update_language_mappings(self, room_id):
#         """Update language mappings for a room."""
#         if room_id not in self.language_mapping:
#             self.language_mapping[room_id] = {}
        
#         # Initialize mapping with all available languages
#         self.language_mapping[room_id] = {lang_code: [] for lang_code in list(languages.values())}
        
#         # Get all users in the room
#         room_users = {uid: details for uid, details in user_session_details.items() 
#                      if details.get("room_id") == room_id}
        
#         # Update mappings based on user language preferences
#         for recipient_id, details in room_users.items():
#             tgt_lang = details.get("language", "Unknown")
#             if tgt_lang in languages:
#                 tgt_lang_code = languages[tgt_lang]
#                 self.language_mapping[room_id][tgt_lang_code].append(recipient_id)
#             else:
#                 print(f"Warning: Unknown language '{tgt_lang}' for user {recipient_id}")

#     def start_threads(self, room_id, start_up=False):
#         """Start processing threads for a room."""
#         print(f"Starting threads for room {room_id}, start_up={start_up}")
        
#         # Reset flags
#         self.stop_thread_flags[room_id] = False
#         self.shutdown_in_progress[room_id] = False
        
#         # Initialize current speaker ID if not set
#         if room_id not in self.current_speaker_id:
#             # Set a default speaker ID or get it from somewhere
#             self.current_speaker_id[room_id] = "default_speaker"
#             print(f"Set default speaker ID for room {room_id}")

#         if start_up:
#             # Update language mappings based on current users in the room
#             self.update_language_mappings(room_id)
            
#             # Initialize room-specific collections if needed
#             if room_id not in self.threads:
#                 self.threads[room_id] = {}
            
#             if room_id not in self.ws_connections:
#                 self.ws_connections[room_id] = {}
            
#             if room_id not in self.ws_locks:
#                 self.ws_locks[room_id] = {}
            
#             if room_id not in self.audio_buffers:
#                 self.audio_buffers[room_id] = {}
#                 self.buffer_locks[room_id] = {}
            
#             # Start a thread for each language that has users
#             for lang_code in self.language_mapping[room_id]:
#                 if len(self.language_mapping[room_id][lang_code]) > 0:
#                     print(f"Instantiating thread for room {room_id}, language {lang_code}")
                    
#                     # Initialize buffers and locks for this language
#                     self.audio_buffers[room_id][lang_code] = np.array([], np.float32)
#                     self.buffer_locks[room_id][lang_code] = threading.Lock()
#                     self.ws_locks[room_id][lang_code] = threading.Lock()
#                     self.ws_connections[room_id][lang_code] = None  # Will be set in process_inference_async

#                     # Create and start the thread
#                     thread = threading.Thread(
#                         target=self.process_inference_wrapper,
#                         args=(room_id, lang_code, lang_code)
#                     )
#                     thread.daemon = True
#                     self.threads[room_id][lang_code] = thread
#                     print(f"Starting thread for room {room_id}, language {lang_code}")
#                     thread.start()

# using nllb to translaste and deliver. 
# class AudioHandler:
#     def __init__(self):
#         # Room-specific data structures
#         self.threads = {}  # {room_id: {lang_code: thread}}
#         self.audio_buffers = {}  # {room_id: {lang_code: buffer}}
#         self.buffer_locks = {}  # {room_id: {lang_code: lock}}
#         self.language_mapping = {}  # {room_id: {lang_code: [user_ids]}}
        
#         self.stop_thread_flags = {}  # {room_id: bool}
#         self.processing_complete = {}  # {room_id: {lang_code: Event}}
#         self.batch_size = AudioConfig.SAMPLERATE * 1
#         self.socketio = None
        
#         self.current_speaker_id = {}
#         self.user_counts = {}  # {room_id: count}
#         self.shutdown_in_progress = {}  # {room_id: bool}
        
#         # WebSocket connections
#         self.ws_connections = {}  # {room_id: {lang_code: websocket}}
#         self.ws_locks = {}  # {room_id: {lang_code: lock}}
        
#         # Event loop for asyncio
#         self.loop = asyncio.new_event_loop()
#         self.loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
#         self.loop_thread.start()
    
#     def _run_event_loop(self):
#         """Run the asyncio event loop in a separate thread."""
#         asyncio.set_event_loop(self.loop)
#         self.loop.run_forever()

#     def handle_audio(self, data, socketio, room_id):
#         """Handle incoming audio data for a specific room."""
#         if not self.socketio:
#             self.socketio = socketio

#         # Don't accept new audio if shutdown is in progress
#         if self.shutdown_in_progress.get(room_id, False):
#             return

#         # Start threads if not already running for this room
#         if room_id not in self.threads:
#             print(f"Starting new threads for room {room_id}")
#             self.start_threads(room_id, start_up=True)
        
#         if room_id in self.current_speaker_id:
#             segment_handler.add_segment(room_id=room_id,
#                                         audio=data,
#                                         speaker=self.current_speaker_id[room_id])

#             # Add audio data to buffers for each language in the room
#             for lang_code in self.language_mapping.get(room_id, {}):
#                 self.add_audio_to_buffer(room_id, lang_code, data)
#         else:
#             print(f"Warning: No current speaker ID set for room {room_id}")

#     async def establish_websocket_connection(self, room_id, lang_code):
#         """Establish WebSocket connection for a specific room and language."""
#         try:
#             print(f"Attempting to establish WebSocket connection for room {room_id}, language {lang_code}")
#             ws = await websockets.connect(ServerConfig.ws_server_url, max_size=1024*1024*10)
            
#             # Initialize WebSocket connection
#             room_user_id = self.get_room_user_id(room_id, lang_code)
            
#             # Send initial message to reset buffers
#             payload = {
#                 "client_id": room_user_id,
#                 "audio_data": [],
#                 "sampling_rate": AudioConfig.SAMPLERATE,
#                 "tgt_lang": lang_code,
#                 "reset_buffers": "true"
#             }
#             await ws.send(json.dumps(payload))
#             print(f"Initial reset message sent for {room_user_id}")
            
#             # Wait for initial response
#             response = await ws.recv()
#             print(f"WebSocket connection established for room {room_id}, language {lang_code}")
            
#             return ws
#         except Exception as e:
#             print(f"Error establishing WebSocket connection for room {room_id}, language {lang_code}: {e}")
#             return None

#     async def send_audio_via_websocket(self, ws, room_user_id, audio_samples, tgt_lang, reset_buffers="false"):
#         """Send audio data via WebSocket and receive response."""
#         if ws is None:
#             print(f"Cannot send audio: WebSocket connection is None for {room_user_id}")
#             return None
            
#         try:
#             payload = {
#                 "client_id": room_user_id,
#                 "audio_data": audio_samples.tolist(),
#                 "sampling_rate": AudioConfig.SAMPLERATE,
#                 "tgt_lang": tgt_lang,
#                 "reset_buffers": reset_buffers
#             }
            
#             await ws.send(json.dumps(payload))
#             response = await ws.recv()
#             return json.loads(response)
#         except Exception as e:
#             print(f"Error sending audio via WebSocket for {room_user_id}: {e}")
#             return None

#     async def process_inference_async(self, room_id, lang_code, tgt_lang):
#         """Async version of process_inference using WebSockets."""
#         global user_sockets
        
#         print(f"Starting WebSocket inference thread for room {room_id}, language {lang_code}")
        
#         # Check if we should stop before even starting
#         if self.stop_thread_flags.get(room_id, False):
#             print(f"Thread for {room_id}/{lang_code} asked to stop before starting")
#             return
        
#         try:
#             # Establish WebSocket connection
#             ws = await self.establish_websocket_connection(room_id, lang_code)
#             if not ws:
#                 print(f"Failed to establish WebSocket connection for room {room_id}, language {lang_code}")
#                 return
            
#             # Store the websocket connection - use locks to prevent race conditions
#             if room_id in self.ws_locks and lang_code in self.ws_locks[room_id]:
#                 with self.ws_locks[room_id][lang_code]:
#                     if room_id in self.ws_connections and lang_code in self.ws_connections[room_id]:
#                         self.ws_connections[room_id][lang_code] = ws
            
#             while not self.stop_thread_flags.get(room_id, False):
#                 # Check if the connection is still valid
#                 if ws.close_code is not None:
#                     print(f"WebSocket closed unexpectedly for {room_id}/{lang_code}, attempting to reconnect")
#                     ws = await self.establish_websocket_connection(room_id, lang_code)
#                     if not ws:
#                         print(f"Failed to re-establish WebSocket connection for {room_id}/{lang_code}")
#                         break
                    
#                     with self.ws_locks[room_id][lang_code]:
#                         self.ws_connections[room_id][lang_code] = ws
                
#                 samples = np.array([], np.float32)
                
#                 # Process audio samples if available
#                 if room_id in self.buffer_locks and lang_code in self.buffer_locks[room_id]:
#                     with self.buffer_locks[room_id][lang_code]:
#                         if room_id in self.audio_buffers and lang_code in self.audio_buffers[room_id]:
#                             buffer = self.audio_buffers[room_id][lang_code]
#                             if len(buffer) > 0:
#                                 # Process in smaller chunks to ensure smooth handling
#                                 chunk_size = min(self.batch_size, len(buffer))
#                                 samples = buffer[:chunk_size]
#                                 self.audio_buffers[room_id][lang_code] = buffer[chunk_size:]
                
#                 if len(samples) > 0:
#                     recipient_ids = []
#                     if room_id in self.language_mapping and lang_code in self.language_mapping[room_id]:
#                         recipient_ids = self.language_mapping[room_id].get(lang_code, [])
                    
#                     room_user_id = self.get_room_user_id(room_id, lang_code)
                    
#                     # Process the audio chunk using WebSocket
#                     if room_id in self.ws_locks and lang_code in self.ws_locks[room_id]:
#                         with self.ws_locks[room_id][lang_code]:
#                             if room_id in self.ws_connections and lang_code in self.ws_connections[room_id]:
#                                 response = await self.send_audio_via_websocket(
#                                     self.ws_connections[room_id][lang_code],
#                                     room_user_id,
#                                     samples,
#                                     tgt_lang
#                                 )
                                
#                                 if response:
#                                     self.handle_inference_response_ws(response, room_id, lang_code, recipient_ids)
                    
 
            
#             # Clean up WebSocket connection
#             print(f"Closing WebSocket for {room_id}/{lang_code}")
#             if ws.close_code is not None:
#                 await ws.close()
            
#             # Signal completion before exiting
#             if room_id in self.processing_complete and lang_code in self.processing_complete[room_id]:
#                 print(f"Setting completion event for room {room_id}, language {lang_code}")
#                 self.processing_complete[room_id][lang_code].set()
                
#         except Exception as e:
#             print(f"Exception in process_inference_async for room {room_id}, language {lang_code}: {e}")
#             traceback.print_exc()
#             if room_id in self.processing_complete and lang_code in self.processing_complete[room_id]:
#                 self.processing_complete[room_id][lang_code].set()

#     def process_inference_wrapper(self, room_id, lang_code, tgt_lang):
#         """Wrapper to run async process_inference in a sync context."""
#         future = asyncio.run_coroutine_threadsafe(
#             self.process_inference_async(room_id, lang_code, tgt_lang),
#             self.loop
#         )
        
#         # Handle any exceptions from the future
#         try:
#             future.result()
#         except Exception as e:
#             print(f"Error in process_inference_wrapper: {e}")
#             traceback.print_exc()

#     async def stop_websocket_connections(self, room_id):
#         """Close all WebSocket connections for a room."""
#         if room_id in self.ws_connections:
#             for lang_code, ws in list(self.ws_connections[room_id].items()):
#                 try:
#                     if ws and  ws.close_code  is not None:
#                         # Send final cleanup message
#                         room_user_id = self.get_room_user_id(room_id, lang_code)
#                         await self.send_audio_via_websocket(
#                             ws, 
#                             room_user_id, 
#                             np.array([], np.float32), 
#                             "eng", 
#                             reset_buffers="true"
#                         )
#                         await ws.close()
#                 except Exception as e:
#                     print(f"Error closing WebSocket for {room_id}, {lang_code}: {e}")

#     def stop_threads(self, room_id=None):
#         """Stop threads gracefully, ensuring all buffered audio is processed."""
#         if room_id:
#             print(f"Initiating shutdown for room {room_id}")
            
#             # First check if the room exists in our data structures
#             if room_id not in self.threads:
#                 print(f"No threads found for room {room_id}, skipping shutdown")
#                 return
                
#             # Log audio buffer status
#             if room_id in self.audio_buffers:
#                 for lang_code, buffer in self.audio_buffers[room_id].items():
#                     print(f"{lang_code} has {len(buffer)/16000} seconds of unprocessed audio during shut down")
            
#             # Mark shutdown in progress to prevent new audio from being added
#             self.shutdown_in_progress[room_id] = True
            
#             # Set stop flag for the room to signal threads to terminate
#             self.stop_thread_flags[room_id] = True          
            
#             # Initialize completion events for each language
#             self.processing_complete[room_id] = {
#                 lang_code: threading.Event()
#                 for lang_code in self.threads.get(room_id, {}).keys()
#             }
            
#             # Wait for all threads to finish processing remaining audio
#             print(f"Waiting for processing completion in room {room_id}")
#             for lang_code, event in self.processing_complete[room_id].items():
#                 # Longer timeout to ensure processing completes
#                 if not event.wait(timeout=10.0):
#                     print(f"Warning: Timeout waiting for {lang_code} processing in room {room_id}")
#             print(f"All processes have completed. Initializing shutdown.")  
            
#             # Close all WebSocket connections
#             future = asyncio.run_coroutine_threadsafe(
#                 self.stop_websocket_connections(room_id),
#                 self.loop
#             )
            
#             # Wait for the future to complete with a timeout
#             try:
#                 future.result(timeout=5.0)
#             except Exception as e:
#                 print(f"Error stopping WebSocket connections: {e}")
            
#             # Check remaining audio after processing
#             if room_id in self.audio_buffers:
#                 for lang_code, buffer in self.audio_buffers[room_id].items():
#                     print(f"{lang_code} has {len(buffer)/16000} seconds of unprocessed audio after shut down")
            
#             # Clean up room resources
#             print(f"Cleaning up resources for room {room_id}")
            
#             # Clean up room-specific data structures
#             if room_id in self.threads:
#                 # Stop all threads first
#                 for thread in self.threads[room_id].values():
#                     if thread.is_alive():
#                         thread.join(timeout=2.0)
                
#                 # Now clean up the data structures
#                 self.threads.pop(room_id, None)
#                 self.audio_buffers.pop(room_id, None)
#                 self.buffer_locks.pop(room_id, None)
#                 self.language_mapping.pop(room_id, None)
#                 self.processing_complete.pop(room_id, None)
#                 self.shutdown_in_progress.pop(room_id, None)
#                 self.ws_connections.pop(room_id, None)
#                 self.ws_locks.pop(room_id, None)
#                 self.stop_thread_flags.pop(room_id, None)
#                 self.current_speaker_id.pop(room_id, None)
                
#             print(f"Shutdown complete for room {room_id}")
#         else:
#             # Stop all threads across all rooms
#             for room_id in list(self.threads.keys()):
#                 self.stop_threads(room_id)

#     # Helper methods
#     def get_room_user_id(self, room_id, user_id):
#         return f"{room_id}_{user_id}"

#     def has_remaining_audio(self, room_id, lang_code):
#         """Check if there is remaining audio for processing."""
#         if room_id in self.buffer_locks and lang_code in self.buffer_locks[room_id]:
#             with self.buffer_locks[room_id][lang_code]:
#                 if room_id in self.audio_buffers and lang_code in self.audio_buffers[room_id]:
#                     return len(self.audio_buffers[room_id][lang_code]) > 0
#         return False

#     def add_audio_to_buffer(self, room_id, lang_code, audio):
#         """Add audio data to the buffer for processing."""
#         if room_id not in self.buffer_locks:
#             self.buffer_locks[room_id] = {}
#             self.audio_buffers[room_id] = {}
            
#         if lang_code not in self.buffer_locks[room_id]:
#             self.buffer_locks[room_id][lang_code] = threading.Lock()
#             self.audio_buffers[room_id][lang_code] = np.array([], np.float32)
            
#         tensor_samples = torch.tensor(audio, dtype=torch.float32)
#         audio = torchaudio.functional.resample(tensor_samples, orig_freq=48000, new_freq=16_000).cpu().numpy()
        
#         with self.buffer_locks[room_id][lang_code]:
#             if room_id in self.language_mapping and lang_code in self.language_mapping[room_id]:
#                 if len(self.language_mapping[room_id][lang_code]) > 0:
#                     self.audio_buffers[room_id][lang_code] = np.append(self.audio_buffers[room_id][lang_code], audio)

#     def handle_inference_response_ws(self, response, room_id, lang_code, recipient_ids):
#         """Handle inference response from WebSocket."""
#         try:
#             transcriptions = response.get('transcriptions', "")
#             if transcriptions:
#                 text, audio = eval(transcriptions)
#                 print(text)
#                 if text[0] != "[" and get_unique_words_percentage(text) > 0.4:
#                     self.broadcast_results(room_id, lang_code, text, audio, recipient_ids)
#         except Exception as e:
#             print(f"Error handling WebSocket inference response: {e}")

#     def broadcast_results(self, room_id, lang_code, text, audio, recipient_ids):
#         """Broadcast translation results to recipients."""
#         for recipient_id in recipient_ids:
#             target_socket_id = user_sockets.get(recipient_id)
            
             
#             try:
#                 user_lang = self.resolve_user_language(recipient_id)
#                 print("user language selection if ", user_lang)
#                 if user_lang!= "eng":
#                     nll_leng_code = seamless_nllb_lang_mapping.get(user_lang)
#                     text = nllb.translate(text, nllb_languages.English, nll_leng_code)
                    
                    
#                 # Only send audio to recipients who aren't the speaker
#                 if recipient_id != self.current_speaker_id.get(room_id, "unknown"):
#                     # Broadcast audio
#                     audio_np = np.array(audio, dtype=np.float32)
#                     audio_file = f"{room_id}_{lang_code}_translated_audio.wav"
#                     sf.write(audio_file, audio_np, 16000)
                    
#                     with open(audio_file, "rb") as f:
#                         self.socketio.emit(
#                             "receive_audio",
#                             {"audio": f.read()},
#                             room=target_socket_id
#                         )
                
#                 # Broadcast text to all recipients
#                 print(f"inside broadcast {text}")
#                 self.socketio.emit(
#                     "transcription",
#                     {
#                         "english": "",
#                         "translated": text,
#                         "sender_user_id": self.current_speaker_id.get(room_id)
#                     },
#                     to=target_socket_id,
#                 )
#             except Exception as e:
#                 print(f"Broadcast error: {e}")
                
                
#     def resolve_user_language(self, user_id):
#         user_details = user_session_details.get(user_id, None)
#         tgt_lang = user_details.get("language", "unknown")
#         tgt_lang_code = languages.get(tgt_lang, "eng")
#         return tgt_lang_code

#     def update_language_mappings(self, room_id):
#         """Update language mappings for a room."""
#         if room_id not in self.language_mapping:
#             self.language_mapping[room_id] = {}
        
#         # Initialize mapping with all available languages
#         self.language_mapping[room_id] = {lang_code: [] for lang_code in list(languages.values())}
        
#         # Get all users in the room
#         room_users = {uid: details for uid, details in user_session_details.items() 
#                      if details.get("room_id") == room_id}
        
#         # Update mappings based on user language preferences
#         for recipient_id, details in room_users.items():
#             tgt_lang = details.get("language", "Unknown")
#             if tgt_lang in languages:
#                 tgt_lang_code = languages[tgt_lang]
#                 tgt_lang_code = "eng"
#                 self.language_mapping[room_id][tgt_lang_code].append(recipient_id)
                
#                 print("\n"*5, f"updating languages: {self.language_mapping}", "\n"*5)
#             else:
#                 print(f"Warning: Unknown language '{tgt_lang}' for user {recipient_id}")

#     def start_threads(self, room_id, start_up=False):
#         """Start processing threads for a room."""
#         print(f"Starting threads for room {room_id}, start_up={start_up}")
        
#         # Reset flags
#         self.stop_thread_flags[room_id] = False
#         self.shutdown_in_progress[room_id] = False
        
#         # Initialize current speaker ID if not set
#         if room_id not in self.current_speaker_id:
#             # Set a default speaker ID or get it from somewhere
#             self.current_speaker_id[room_id] = "default_speaker"
#             print(f"Set default speaker ID for room {room_id}")

#         if start_up:
#             # Update language mappings based on current users in the room
#             self.update_language_mappings(room_id)
            
#             # Initialize room-specific collections if needed
#             if room_id not in self.threads:
#                 self.threads[room_id] = {}
            
#             if room_id not in self.ws_connections:
#                 self.ws_connections[room_id] = {}
            
#             if room_id not in self.ws_locks:
#                 self.ws_locks[room_id] = {}
            
#             if room_id not in self.audio_buffers:
#                 self.audio_buffers[room_id] = {}
#                 self.buffer_locks[room_id] = {}
            
#             # Start a thread for each language that has users
#             for lang_code in self.language_mapping[room_id]:
#                 if len(self.language_mapping[room_id][lang_code]) > 0:
#                     print(f"Instantiating thread for room {room_id}, language {lang_code}")
                    
#                     # Initialize buffers and locks for this language
#                     self.audio_buffers[room_id][lang_code] = np.array([], np.float32)
#                     self.buffer_locks[room_id][lang_code] = threading.Lock()
#                     self.ws_locks[room_id][lang_code] = threading.Lock()
#                     self.ws_connections[room_id][lang_code] = None  # Will be set in process_inference_async

#                     # Create and start the thread
#                     thread = threading.Thread(
#                         target=self.process_inference_wrapper,
#                         args=(room_id, lang_code, lang_code)
#                     )
#                     thread.daemon = True
#                     self.threads[room_id][lang_code] = thread
#                     print(f"Starting thread for room {room_id}, language {lang_code}")
#                     thread.start()

# This iteration properly end the websockets and closes the socket,
# but gives only silenec when turned on again.
# class AudioHandler:
#     def __init__(self):
#         # Room-specific data structures
#         self.threads = {}  # {room_id: {lang_code: thread}}
#         self.audio_buffers = {}  # {room_id: {lang_code: buffer}}
#         self.buffer_locks = {}  # {room_id: {lang_code: lock}}
#         self.language_mapping = {}  # {room_id: {lang_code: [user_ids]}}
        
#         self.stop_thread_flags = {}  # {room_id: bool}
#         self.processing_complete = {}  # {room_id: {lang_code: Event}}
#         self.batch_size = AudioConfig.SAMPLERATE * 1
#         self.socketio = None
        
#         self.current_speaker_id = {}
#         self.user_counts = {}  # {room_id: count}
#         self.shutdown_in_progress = {}  # {room_id: bool}
        
#         # WebSocket connections
#         self.ws_connections = {}  # {room_id: {lang_code: websocket}}
#         self.ws_locks = {}  # {room_id: {lang_code: lock}}
        
#         # Event loop for asyncio
#         self.loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(self.loop)

#     def handle_audio(self, data, socketio, room_id):
#         """Handle incoming audio data for a specific room."""
#         if not self.socketio:
#             self.socketio = socketio

#         # Don't accept new audio if shutdown is in progress
#         if self.shutdown_in_progress.get(room_id, False):
#             return

#         # Start threads if not already running for this room
#         if room_id not in self.threads:
#             self.start_threads(room_id, start_up=True)

#         segment_handler.add_segment(room_id=room_id,
#                                     audio=data,
#                                     speaker=self.current_speaker_id[room_id])

#         # Add audio data to buffers for each language in the room
#         for lang_code in self.language_mapping.get(room_id, {}):
#             self.add_audio_to_buffer(room_id, lang_code, data)

#     async def establish_websocket_connection(self, room_id, lang_code):
#         """Establish WebSocket connection for a specific room and language."""
#         try:
#             ws = await websockets.connect(ServerConfig.ws_server_url, max_size=1024*1024*10)
            
#             # Initialize WebSocket connection
#             room_user_id = self.get_room_user_id(room_id, lang_code)
            
#             # Send initial message to reset buffers
#             payload = {
#                 "client_id": room_user_id,
#                 "audio_data": [],
#                 "sampling_rate": AudioConfig.SAMPLERATE,
#                 "tgt_lang": lang_code,
#                 "reset_buffers": "true"
#             }
#             await ws.send(json.dumps(payload))
#             print(f"sending audio for {room_user_id}")
            
#             # Wait for initial response
#             response = await ws.recv()
#             print(f"WebSocket connection established for room {room_id}, language {lang_code}")
            
#             return ws
#         except Exception as e:
#             print(f"Error establishing WebSocket connection for room {room_id}, language {lang_code}: {e}")
#             return None

#     async def send_audio_via_websocket(self, ws, room_user_id, audio_samples, tgt_lang, reset_buffers="false"):
#         """Send audio data via WebSocket and receive response."""
#         try:
#             payload = {
#                 "client_id": room_user_id,
#                 "audio_data": audio_samples.tolist(),
#                 "sampling_rate": AudioConfig.SAMPLERATE,
#                 "tgt_lang": tgt_lang,
#                 "reset_buffers": reset_buffers
#             }
            
#             await ws.send(json.dumps(payload))
#             response = await ws.recv()
#             return json.loads(response)
#         except Exception as e:
#             print(f"Error sending audio via WebSocket: {e}")
#             return None

#     async def process_inference_async(self, room_id, lang_code, tgt_lang):
#         """Async version of process_inference using WebSockets."""
#         global user_sockets
        
#         try:
#             print(f"Starting WebSocket inference thread for room {room_id}, language {lang_code}")
            
#             # Establish WebSocket connection
#             ws = await self.establish_websocket_connection(room_id, lang_code)
#             if not ws:
#                 print(f"Failed to establish WebSocket connection for room {room_id}, language {lang_code}")
#                 return
            
#             self.ws_connections[room_id][lang_code] = ws
            
#             while True:
#                 # Check if we should stop and if there's no more audio
#                 if (self.stop_thread_flags.get(room_id, False) and 
#                     not self.has_remaining_audio(room_id, lang_code)):
#                     print(f"Stopping inference for room {room_id}, language {lang_code}")
#                     break
                
#                 samples = np.array([], np.float32)
                
#                 with self.buffer_locks[room_id][lang_code]:
#                     if len(self.audio_buffers[room_id][lang_code]) > 0:
#                         # Process in smaller chunks to ensure smooth handling
#                         chunk_size = min(self.batch_size, len(self.audio_buffers[room_id][lang_code]))
#                         samples = self.audio_buffers[room_id][lang_code][:chunk_size]
#                         self.audio_buffers[room_id][lang_code] = self.audio_buffers[room_id][lang_code][chunk_size:]
                
#                 if len(samples) > 0:
#                     recipient_ids = self.language_mapping[room_id].get(lang_code, [])
#                     room_user_id = self.get_room_user_id(room_id, lang_code)
                    
#                     # Process the audio chunk using WebSocket
#                     with self.ws_locks[room_id][lang_code]:
#                         response = await self.send_audio_via_websocket(
#                             self.ws_connections[room_id][lang_code],
#                             room_user_id,
#                             samples,
#                             tgt_lang
#                         )
                        
#                     if response:
#                         self.handle_inference_response_ws(response, room_id, lang_code, recipient_ids)
                        
#                         # Small delay to prevent overwhelming the system
#                         await asyncio.sleep(0.01)
#                 else:
#                     # Smaller sleep when no audio to process
#                     await asyncio.sleep(0.05)
            
#             # Close WebSocket connection
#             await ws.close()
            
#             # Signal completion before exiting
#             if room_id in self.processing_complete and lang_code in self.processing_complete[room_id]:
#                 print(f"Setting completion event for room {room_id}, language {lang_code}")
#                 self.processing_complete[room_id][lang_code].set()
                
#         except Exception as e:
#             print(f"Exception in process_inference_async for room {room_id}, language {lang_code}: {e}")
#             if room_id in self.processing_complete and lang_code in self.processing_complete[room_id]:
#                 self.processing_complete[room_id][lang_code].set()

#     def process_inference_wrapper(self, room_id, lang_code, tgt_lang):
#         """Wrapper to run async process_inference in a sync context."""
#         asyncio.run_coroutine_threadsafe(
#             self.process_inference_async(room_id, lang_code, tgt_lang),
#             self.loop
#         )

#     async def stop_websocket_connections(self, room_id):
#         """Close all WebSocket connections for a room."""
#         if room_id in self.ws_connections:
#             for lang_code, ws in self.ws_connections[room_id].items():
#                 try:
#                     # Send final cleanup message
#                     room_user_id = self.get_room_user_id(room_id, lang_code)
#                     await self.send_audio_via_websocket(
#                         ws, 
#                         room_user_id, 
#                         np.array([], np.float32), 
#                         "eng", 
#                         reset_buffers="true"
#                     )
#                     await ws.close()
#                 except Exception as e:
#                     print(f"Error closing WebSocket for {room_id}, {lang_code}: {e}")

#     def stop_threads(self, room_id=None):
#         """Stop threads gracefully, ensuring all buffered audio is processed."""
#         if room_id:
#             print(f"Initiating shutdown for room {room_id}")
#             for lang_code, buffer in self.audio_buffers[room_id].items():
#                 print(f"{lang_code} has {len(buffer)/16000} seconds of unprocessed audio during shut down")
            
#             # Mark shutdown in progress
#             self.shutdown_in_progress[room_id] = True
            
#             # Set stop flag for the room
#             self.stop_thread_flags[room_id] = True          
            
#             # Initialize completion events for each language
#             self.processing_complete[room_id] = {
#                 lang_code: threading.Event()
#                 for lang_code in self.threads.get(room_id, {}).keys()
#             }
            
#             # Wait for all threads to finish processing remaining audio
#             print(f"Waiting for processing completion in room {room_id}")
#             for lang_code, event in self.processing_complete[room_id].items():
#                 # Longer timeout to ensure processing completes
#                 if not event.wait(timeout=10.0):
#                     print(f"Warning: Timeout waiting for {lang_code} processing in room {room_id}")
#             print(f"All process have completed. Initializing shutdown.")  
            
#             # Close all WebSocket connections
#             asyncio.run_coroutine_threadsafe(
#                 self.stop_websocket_connections(room_id),
#                 self.loop
#             ).result(timeout=5.0)
            
#             for lang_code, buffer in self.audio_buffers[room_id].items():
#                 print(f"{lang_code} has {len(buffer)/16000} seconds of unprocessed audio during shut down")
            
#             # Clean up room resources
#             print(f"Cleaning up resources for room {room_id}")
            
#             # Clean up room-specific data structures
#             if room_id in self.threads:
#                 for thread in self.threads[room_id].values():
#                     thread.join(timeout=2.0)
                
#                 del self.threads[room_id]
#                 del self.audio_buffers[room_id]
#                 del self.buffer_locks[room_id]
#                 del self.language_mapping[room_id]
#                 del self.processing_complete[room_id]
#                 del self.shutdown_in_progress[room_id]
#                 del self.ws_connections[room_id]
#                 del self.ws_locks[room_id]
                
#             print(f"Shutdown complete for room {room_id}")
#         else:
#             # Stop all threads across all rooms
#             for room_id in list(self.threads.keys()):
#                 self.stop_threads(room_id)

#     # Helper methods
#     def get_room_user_id(self, room_id, user_id):
#         return f"{room_id}_{user_id}"

#     def has_remaining_audio(self, room_id, lang_code):
#         with self.buffer_locks[room_id][lang_code]:
#             return len(self.audio_buffers[room_id][lang_code]) > 0

#     def add_audio_to_buffer(self, room_id, lang_code, audio):
#         if room_id not in self.buffer_locks:
#             self.buffer_locks[room_id] = {}
#             self.audio_buffers[room_id] = {}
            
#         if lang_code not in self.buffer_locks[room_id]:
#             self.buffer_locks[room_id][lang_code] = threading.Lock()
#             self.audio_buffers[room_id][lang_code] = np.array([], np.float32)
            
#         tensor_samples = torch.tensor(audio, dtype=torch.float32)
#         audio = torchaudio.functional.resample(tensor_samples, orig_freq=48000, new_freq=16_000).cpu().numpy()
        
#         with self.buffer_locks[room_id][lang_code]:
#             if len(self.language_mapping[room_id][lang_code]) > 0:
#                 self.audio_buffers[room_id][lang_code] = np.append(self.audio_buffers[room_id][lang_code], audio)

#     def handle_inference_response_ws(self, response, room_id, lang_code, recipient_ids):
#         """Handle inference response from WebSocket."""
#         try:
#             transcriptions = response.get('transcriptions', "")
#             if transcriptions:
#                 text, audio = eval(transcriptions)
#                 if text[0] != "[" and get_unique_words_percentage(text) > 0.4:
#                     self.broadcast_results(room_id, lang_code, text, audio, recipient_ids)
#         except Exception as e:
#             print(f"Error handling WebSocket inference response: {e}")

#     def broadcast_results(self, room_id, lang_code, text, audio, recipient_ids):
#         for recipient_id in recipient_ids:
#             target_socket_id = user_sockets.get(recipient_id)
#             try:
#                 if recipient_id != self.current_speaker_id.get(room_id, "unknown"):
#                     # Broadcast audio
#                     audio_np = np.array(audio, dtype=np.float32)
#                     audio_file = f"{room_id}_{lang_code}_translated_audio.wav"
#                     sf.write(audio_file, audio_np, 16000)
                    
#                     with open(audio_file, "rb") as f:
#                         self.socketio.emit(
#                             "receive_audio",
#                             {"audio": f.read()},
#                             room=target_socket_id
#                         )
                
#                 # Broadcast text
#                 self.socketio.emit(
#                     "transcription",
#                     {
#                         "english": "",
#                         "translated": text,
#                         "sender_user_id": self.current_speaker_id.get(room_id)
#                     },
#                     to=target_socket_id,
#                 )
#             except Exception as e:
#                 print(f"Broadcast error: {e}")

#     def update_language_mappings(self, room_id):
#         if room_id not in self.language_mapping:
#             self.language_mapping[room_id] = {}
        
#         self.language_mapping[room_id] = {lang_code: [] for lang_code in list(languages.values())}
        
#         room_users = {uid: details for uid, details in user_session_details.items() 
#                      if details.get("room_id") == room_id}
        
#         for recipient_id, details in room_users.items():
#             tgt_lang = details.get("language", "Unknown")
#             tgt_lang_code = languages[tgt_lang]
#             self.language_mapping[room_id][tgt_lang_code].append(recipient_id)

#     def start_threads(self, room_id, start_up=False):
#         self.stop_thread_flags[room_id] = False
#         self.shutdown_in_progress[room_id] = False

#         if start_up:
#             self.update_language_mappings(room_id)
            
#             if room_id not in self.threads:
#                 self.threads[room_id] = {}
                
#             if room_id not in self.ws_connections:
#                 self.ws_connections[room_id] = {}
#                 self.ws_locks[room_id] = {}
            
#             for lang_code in self.language_mapping[room_id]:
#                 if len(self.language_mapping[room_id][lang_code]) > 0:
#                     print(f"Instantiating thread for room {room_id}, language {lang_code}")
                    
#                     # Initialize room-specific buffers and locks
#                     if room_id not in self.audio_buffers:
#                         self.audio_buffers[room_id] = {}
#                         self.buffer_locks[room_id] = {}
                    
#                     self.audio_buffers[room_id][lang_code] = np.array([], np.float32)
#                     self.buffer_locks[room_id][lang_code] = threading.Lock()
#                     self.ws_locks[room_id][lang_code] = threading.Lock()

#                     # Create and start the thread
#                     thread = threading.Thread(
#                         target=self.process_inference_wrapper,
#                         args=(room_id, lang_code, lang_code)
#                     )
#                     thread.daemon = True
#                     self.threads[room_id][lang_code] = thread
#                     print(f"Starting thread for room {room_id}, language {lang_code}")
#                     thread.start()

# This iteration is one with single transcription thread,
# and multiple traslation threads, requests ws_server,
# for translation too
# class AudioHandler:
#     def __init__(self):
#         # Room-specific data structures
#         self.threads = {}  # {room_id: {thread_type: thread}}
#         self.audio_buffer = {}  # {room_id: buffer} - Shared audio buffer for each room
#         self.audio_buffer_locks = {}  # {room_id: lock}
        
#         # Transcription results buffer
#         self.transcription_buffer = {}  # {room_id: {text, audio}} - Intermediate buffer for English transcriptions
#         self.transcription_buffer_locks = {}  # {room_id: lock}
        
#         # Language-specific translation buffers
#         self.translation_buffers = {}  # {room_id: {lang_code: [{text, audio}]}}
#         self.translation_buffer_locks = {}  # {room_id: {lang_code: lock}}
        
#         self.language_mapping = {}  # {room_id: {lang_code: [user_ids]}}
        
#         self.stop_thread_flags = {}  # {room_id: bool}
#         self.processing_complete = {}  # {room_id: {thread_type: Event}}
#         self.batch_size = AudioConfig.SAMPLERATE * 1
#         self.socketio = None
        
#         self.current_speaker_id = {}
#         self.user_counts = {}  # {room_id: count}
#         self.shutdown_in_progress = {}  # {room_id: bool}
        
#         # WebSocket connections
#         self.transcription_ws = {}  # {room_id: websocket} - Single websocket for English transcription
#         self.transcription_ws_locks = {}  # {room_id: lock}
        
#         self.translation_ws = {}  # {room_id: {lang_code: websocket}}
#         self.translation_ws_locks = {}  # {room_id: {lang_code: lock}}
        
#         # Event loop for asyncio
#         self.loop = asyncio.new_event_loop()
#         self.loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
#         self.loop_thread.start()
    
#     def _run_event_loop(self):
#         """Run the asyncio event loop in a separate thread."""
#         asyncio.set_event_loop(self.loop)
#         self.loop.run_forever()

#     def handle_audio(self, data, socketio, room_id):
#         """Handle incoming audio data for a specific room by adding to shared buffer."""
#         if not self.socketio:
#             self.socketio = socketio

#         # Don't accept new audio if shutdown is in progress
#         if self.shutdown_in_progress.get(room_id, False):
#             return

#         # Start threads if not already running for this room
#         if room_id not in self.threads:
#             print(f"Starting new threads for room {room_id}")
#             self.start_threads(room_id, start_up=True)
        
#         if room_id in self.current_speaker_id:
#             segment_handler.add_segment(room_id=room_id,
#                                         audio=data,
#                                         speaker=self.current_speaker_id[room_id])

#             # Add audio data to shared buffer for the room
#             self.add_audio_to_shared_buffer(room_id, data)
#         else:
#             print(f"Warning: No current speaker ID set for room {room_id}")

#     def add_audio_to_shared_buffer(self, room_id, audio):
#         """Add audio data to the shared buffer for processing."""
#         if room_id not in self.audio_buffer_locks:
#             self.audio_buffer_locks[room_id] = threading.Lock()
#             self.audio_buffer[room_id] = np.array([], np.float32)
            
#         tensor_samples = torch.tensor(audio, dtype=torch.float32)
#         resampled_audio = torchaudio.functional.resample(tensor_samples, orig_freq=48000, new_freq=16_000).cpu().numpy()
        
#         with self.audio_buffer_locks[room_id]:
#             self.audio_buffer[room_id] = np.append(self.audio_buffer[room_id], resampled_audio)

#     async def establish_transcription_websocket(self, room_id):
#         """Establish WebSocket connection for English transcription."""
#         try:
#             print(f"Establishing transcription WebSocket connection for room {room_id}")
#             ws = await websockets.connect(ServerConfig.ws_server_url, max_size=1024*1024*10)
            
#             # Initialize WebSocket connection
#             room_user_id = self.get_room_user_id(room_id, "eng")
            
#             # Send initial message to reset buffers
#             payload = {
#                 "client_id": room_user_id,
#                 "audio_data": [],
#                 "sampling_rate": AudioConfig.SAMPLERATE,
#                 "tgt_lang": "eng",  # Always English for transcription
#                 "reset_buffers": "true"
#             }
#             await ws.send(json.dumps(payload))
#             print(f"Initial reset message sent for transcription {room_user_id}")
            
#             # Wait for initial response
#             response = await ws.recv()
#             print(f"Transcription WebSocket connection established for room {room_id}")
            
#             return ws
#         except Exception as e:
#             print(f"Error establishing transcription WebSocket connection for room {room_id}: {e}")
#             return None

#     async def establish_translation_websocket(self, room_id, lang_code):
#         """Establish WebSocket connection for translation to a specific language."""
#         # Skip establishing connection for English as it won't need translation
#         if lang_code == "eng":
#             return None
            
#         try:
#             print(f"Establishing translation WebSocket connection for room {room_id}, language {lang_code}")
#             ws = await websockets.connect(ServerConfig.ws_server_url, max_size=1024*1024*10)
            
#             # Initialize WebSocket connection
#             room_user_id = self.get_room_user_id(room_id, lang_code)
            
#             # Send initial message to reset buffers
#             payload = {
#                 "client_id": room_user_id,
#                 "text_data": "",  # Empty for initial setup
#                 "tgt_lang": lang_code,
#                 "reset_buffers": "true"
#             }
#             await ws.send(json.dumps(payload))
#             print(f"Initial reset message sent for translation {room_user_id}")
            
#             # Wait for initial response
#             response = await ws.recv()
#             print(f"Translation WebSocket connection established for room {room_id}, language {lang_code}")
            
#             return ws
#         except Exception as e:
#             print(f"Error establishing translation WebSocket connection for room {room_id}, language {lang_code}: {e}")
#             return None

#     async def send_audio_for_transcription(self, ws, room_user_id, audio_samples, reset_buffers="false"):
#         """Send audio data via WebSocket for English transcription and receive response."""
#         if ws is None:
#             print(f"Cannot send audio: Transcription WebSocket connection is None for {room_user_id}")
#             return None
            
#         try:
#             payload = {
#                 "client_id": room_user_id,
#                 "audio_data": audio_samples.tolist(),
#                 "sampling_rate": AudioConfig.SAMPLERATE,
#                 "tgt_lang": "eng",  # Always English for transcription
#                 "reset_buffers": reset_buffers
#             }
            
#             await ws.send(json.dumps(payload))
#             response = await ws.recv()
#             return json.loads(response)
#         except Exception as e:
#             print(f"Error sending audio for transcription via WebSocket for {room_user_id}: {e}")
#             return None

#     async def send_text_for_translation(self, ws, room_user_id, text, tgt_lang, reset_buffers="false"):
#         """Send text data via WebSocket for translation and receive response."""
#         if ws is None:
#             print(f"Cannot send text: Translation WebSocket connection is None for {room_user_id}")
#             return None
            
#         try:
#             payload = {
#                 "client_id": room_user_id,
#                 "text_data": text,
#                 "tgt_lang": tgt_lang,
#                 "reset_buffers": reset_buffers
#             }
            
#             await ws.send(json.dumps(payload))
#             response = await ws.recv()
#             return json.loads(response)
#         except Exception as e:
#             print(f"Error sending text for translation via WebSocket for {room_user_id}: {e}")
#             return None

#     def add_to_transcription_buffer(self, room_id, text, audio):
#         """Add transcription result to the buffer."""
#         if room_id not in self.transcription_buffer_locks:
#             self.transcription_buffer_locks[room_id] = threading.Lock()
#             self.transcription_buffer[room_id] = []
            
#         with self.transcription_buffer_locks[room_id]:
#             self.transcription_buffer[room_id].append({"text": text, "audio": audio})

#     def add_to_translation_buffer(self, room_id, lang_code, text, audio):
#         """Add translation result to the language-specific buffer."""
#         if room_id not in self.translation_buffer_locks:
#             self.translation_buffer_locks[room_id] = {}
#             self.translation_buffers[room_id] = {}
            
#         if lang_code not in self.translation_buffer_locks[room_id]:
#             self.translation_buffer_locks[room_id][lang_code] = threading.Lock()
#             self.translation_buffers[room_id][lang_code] = []
            
#         with self.translation_buffer_locks[room_id][lang_code]:
#             self.translation_buffers[room_id][lang_code].append({"text": text, "audio": audio})

#     async def transcription_thread_async(self, room_id):
#         """Process audio for transcription to English."""
#         print(f"Starting transcription thread for room {room_id}")
        
#         # Check if we should stop before even starting
#         if self.stop_thread_flags.get(room_id, False):
#             print(f"Transcription thread for {room_id} asked to stop before starting")
#             return
        
#         try:
#             # Establish WebSocket connection
#             ws = await self.establish_transcription_websocket(room_id)
#             if not ws:
#                 print(f"Failed to establish transcription WebSocket connection for room {room_id}")
#                 return
            
#             # Store the websocket connection
#             with self.transcription_ws_locks[room_id]:
#                 self.transcription_ws[room_id] = ws
            
#             while not self.stop_thread_flags.get(room_id, False):
#                 # Check if the connection is still valid
#                 if ws.close_code is not None:
#                     print(f"Transcription WebSocket closed unexpectedly for {room_id}, attempting to reconnect")
#                     ws = await self.establish_transcription_websocket(room_id)
#                     if not ws:
#                         print(f"Failed to re-establish transcription WebSocket connection for {room_id}")
#                         break
                    
#                     with self.transcription_ws_locks[room_id]:
#                         self.transcription_ws[room_id] = ws
                
#                 samples = np.array([], np.float32)
                
#                 # Process audio samples if available
#                 with self.audio_buffer_locks[room_id]:
#                     buffer = self.audio_buffer[room_id]
#                     if len(buffer) > 0:
#                         # Process in smaller chunks to ensure smooth handling
#                         chunk_size = min(self.batch_size, len(buffer))
#                         samples = buffer[:chunk_size]
#                         self.audio_buffer[room_id] = buffer[chunk_size:]
                
#                 if len(samples) > 0:
#                     room_user_id = self.get_room_user_id(room_id, "eng")
                    
#                     # Process the audio chunk using WebSocket
#                     with self.transcription_ws_locks[room_id]:
#                         response = await self.send_audio_for_transcription(
#                             self.transcription_ws[room_id],
#                             room_user_id,
#                             samples
#                         )
                        
#                         if response:
#                             transcriptions = response.get('transcriptions', "")
#                             if transcriptions:
#                                 try:
#                                     text, audio = eval(transcriptions)
#                                     print(f"Transcription: {text}")
#                                     if text[0] != "[" and get_unique_words_percentage(text) > 0.4:
#                                         # Add to transcription buffer for distribution
#                                         self.add_to_transcription_buffer(room_id, text, audio)

#                                         # For English users, directly deliver the transcription
#                                         if "eng" in self.language_mapping[room_id] and self.language_mapping[room_id]["eng"]:
#                                             self.broadcast_results(room_id, "eng", text, audio, self.language_mapping[room_id]["eng"])
#                                 except Exception as e:
#                                     print(f"Error processing transcription response: {e}")
#                 else:
#                     # If no audio to process, sleep briefly to avoid busy-waiting
#                     await asyncio.sleep(0.1)
            
#             # Clean up WebSocket connection
#             print(f"Closing transcription WebSocket for {room_id}")
#             if ws.close_code is None:
#                 await ws.close()
            
#             # Signal completion before exiting
#             if room_id in self.processing_complete and "transcription" in self.processing_complete[room_id]:
#                 print(f"Setting completion event for transcription thread in room {room_id}")
#                 self.processing_complete[room_id]["transcription"].set()
                
#         except Exception as e:
#             print(f"Exception in transcription_thread_async for room {room_id}: {e}")
#             traceback.print_exc()
#             if room_id in self.processing_complete and "transcription" in self.processing_complete[room_id]:
#                 self.processing_complete[room_id]["transcription"].set()

#     async def translation_thread_async(self, room_id, lang_code):
#         """Process transcription results for translation to a specific language."""
#         # Skip translation for English as it's already transcribed
#         if lang_code == "eng":
#             print(f"Skipping translation thread for English in room {room_id}")
#             return
            
#         print(f"Starting translation thread for room {room_id}, language {lang_code}")
        
#         # Check if we should stop before even starting
#         if self.stop_thread_flags.get(room_id, False):
#             print(f"Translation thread for {room_id}/{lang_code} asked to stop before starting")
#             return
        
#         try:
#             # Establish WebSocket connection
#             ws = await self.establish_translation_websocket(room_id, lang_code)
#             if not ws:
#                 print(f"Failed to establish translation WebSocket connection for room {room_id}, language {lang_code}")
#                 return
            
#             # Store the websocket connection
#             with self.translation_ws_locks[room_id][lang_code]:
#                 self.translation_ws[room_id][lang_code] = ws
            
#             while not self.stop_thread_flags.get(room_id, False):
#                 # Check if the connection is still valid
#                 if ws.close_code is not None:
#                     print(f"Translation WebSocket closed unexpectedly for {room_id}/{lang_code}, attempting to reconnect")
#                     ws = await self.establish_translation_websocket(room_id, lang_code)
#                     if not ws:
#                         print(f"Failed to re-establish translation WebSocket connection for {room_id}/{lang_code}")
#                         break
                    
#                     with self.translation_ws_locks[room_id][lang_code]:
#                         self.translation_ws[room_id][lang_code] = ws
                
#                 # Check for transcription results to translate
#                 items_to_process = []
#                 with self.transcription_buffer_locks[room_id]:
#                     if self.transcription_buffer[room_id]:
#                         items_to_process = self.transcription_buffer[room_id].copy()
#                         self.transcription_buffer[room_id] = []
                
#                 for item in items_to_process:
#                     room_user_id = self.get_room_user_id(room_id, lang_code)
                    
#                     # Send text for translation using WebSocket
#                     with self.translation_ws_locks[room_id][lang_code]:
#                         try:                                
#                             translated_text, translated_audio = f"{lang_code}" + item, []
                                
#                             # Broadcast to users of this language
#                             if lang_code in self.language_mapping[room_id] and self.language_mapping[room_id][lang_code]:
#                                 self.broadcast_results(
#                                     room_id, 
#                                     lang_code, 
#                                     translated_text, 
#                                     translated_audio or item["audio"],  # Use original audio if translation doesn't provide audio
#                                     self.language_mapping[room_id][lang_code]
#                                 )
#                         except Exception as e:
#                             print(f"Error processing translation response: {e}")
                
#                 if not items_to_process:
#                     # If no transcriptions to process, sleep briefly to avoid busy-waiting
#                     await asyncio.sleep(0.1)
            
#             # Clean up WebSocket connection
#             print(f"Closing translation WebSocket for {room_id}/{lang_code}")
#             if ws.close_code is None:
#                 await ws.close()
            
#             # Signal completion before exiting
#             if room_id in self.processing_complete and f"translation_{lang_code}" in self.processing_complete[room_id]:
#                 print(f"Setting completion event for translation thread {lang_code} in room {room_id}")
#                 self.processing_complete[room_id][f"translation_{lang_code}"].set()
                
#         except Exception as e:
#             print(f"Exception in translation_thread_async for room {room_id}, language {lang_code}: {e}")
#             traceback.print_exc()
#             if room_id in self.processing_complete and f"translation_{lang_code}" in self.processing_complete[room_id]:
#                 self.processing_complete[room_id][f"translation_{lang_code}"].set()

#     def transcription_thread_wrapper(self, room_id):
#         """Wrapper to run async transcription thread in a sync context."""
#         future = asyncio.run_coroutine_threadsafe(
#             self.transcription_thread_async(room_id),
#             self.loop
#         )
        
#         # Handle any exceptions from the future
#         try:
#             future.result()
#         except Exception as e:
#             print(f"Error in transcription_thread_wrapper: {e}")
#             traceback.print_exc()

#     def translation_thread_wrapper(self, room_id, lang_code):
#         """Wrapper to run async translation thread in a sync context."""
#         future = asyncio.run_coroutine_threadsafe(
#             self.translation_thread_async(room_id, lang_code),
#             self.loop
#         )
        
#         # Handle any exceptions from the future
#         try:
#             future.result()
#         except Exception as e:
#             print(f"Error in translation_thread_wrapper: {e}")
#             traceback.print_exc()

#     async def stop_websocket_connections(self, room_id):
#         """Close all WebSocket connections for a room."""
#         # Close transcription WebSocket
#         if room_id in self.transcription_ws:
#             try:
#                 ws = self.transcription_ws[room_id]
#                 if ws and ws.close_code is None:
#                     # Send final cleanup message
#                     room_user_id = self.get_room_user_id(room_id, "eng")
#                     await self.send_audio_for_transcription(
#                         ws, 
#                         room_user_id, 
#                         np.array([], np.float32), 
#                         reset_buffers="true"
#                     )
#                     await ws.close()
#             except Exception as e:
#                 print(f"Error closing transcription WebSocket for {room_id}: {e}")
        
#         # Close translation WebSockets
#         if room_id in self.translation_ws:
#             for lang_code, ws in list(self.translation_ws[room_id].items()):
#                 try:
#                     if ws and ws.close_code is None:
#                         # Send final cleanup message
#                         room_user_id = self.get_room_user_id(room_id, lang_code)
#                         await self.send_text_for_translation(
#                             ws, 
#                             room_user_id, 
#                             "", 
#                             lang_code, 
#                             reset_buffers="true"
#                         )
#                         await ws.close()
#                 except Exception as e:
#                     print(f"Error closing translation WebSocket for {room_id}, {lang_code}: {e}")

#     def stop_threads(self, room_id=None):
#         """Stop threads gracefully, ensuring all buffered audio is processed."""
#         if room_id:
#             print(f"Initiating shutdown for room {room_id}")
            
#             # First check if the room exists in our data structures
#             if room_id not in self.threads:
#                 print(f"No threads found for room {room_id}, skipping shutdown")
#                 return
                
#             # Log audio buffer status
#             if room_id in self.audio_buffer:
#                 print(f"Room {room_id} has {len(self.audio_buffer[room_id])/16000} seconds of unprocessed audio during shut down")
            
#             # Mark shutdown in progress to prevent new audio from being added
#             self.shutdown_in_progress[room_id] = True
            
#             # Set stop flag for the room to signal threads to terminate
#             self.stop_thread_flags[room_id] = True          
            
#             # Initialize completion events for each thread
#             self.processing_complete[room_id] = {
#                 "transcription": threading.Event()
#             }
            
#             # Add completion events for translation threads
#             for lang_code in self.language_mapping.get(room_id, {}):
#                 if lang_code != "eng":  # English doesn't have a translation thread
#                     self.processing_complete[room_id][f"translation_{lang_code}"] = threading.Event()
            
#             # Wait for all threads to finish processing remaining audio
#             print(f"Waiting for processing completion in room {room_id}")
#             for thread_type, event in self.processing_complete[room_id].items():
#                 # Longer timeout to ensure processing completes
#                 if not event.wait(timeout=10.0):
#                     print(f"Warning: Timeout waiting for {thread_type} processing in room {room_id}")
#             print(f"All processes have completed. Initializing shutdown.")  
            
#             # Close all WebSocket connections
#             future = asyncio.run_coroutine_threadsafe(
#                 self.stop_websocket_connections(room_id),
#                 self.loop
#             )
            
#             # Wait for the future to complete with a timeout
#             try:
#                 future.result(timeout=5.0)
#             except Exception as e:
#                 print(f"Error stopping WebSocket connections: {e}")
            
#             # Check remaining audio after processing
#             if room_id in self.audio_buffer:
#                 print(f"Room {room_id} has {len(self.audio_buffer[room_id])/16000} seconds of unprocessed audio after shut down")
            
#             # Clean up room resources
#             print(f"Cleaning up resources for room {room_id}")
            
#             # Clean up room-specific data structures
#             if room_id in self.threads:
#                 # Stop all threads first
#                 for thread in self.threads[room_id].values():
#                     if thread.is_alive():
#                         thread.join(timeout=2.0)
                
#                 # Now clean up the data structures
#                 self.threads.pop(room_id, None)
#                 self.audio_buffer.pop(room_id, None)
#                 self.audio_buffer_locks.pop(room_id, None)
#                 self.transcription_buffer.pop(room_id, None)
#                 self.transcription_buffer_locks.pop(room_id, None)
#                 self.translation_buffers.pop(room_id, None)
#                 self.translation_buffer_locks.pop(room_id, None)
#                 self.language_mapping.pop(room_id, None)
#                 self.processing_complete.pop(room_id, None)
#                 self.shutdown_in_progress.pop(room_id, None)
#                 self.transcription_ws.pop(room_id, None)
#                 self.transcription_ws_locks.pop(room_id, None)
#                 self.translation_ws.pop(room_id, None)
#                 self.translation_ws_locks.pop(room_id, None)
#                 self.stop_thread_flags.pop(room_id, None)
#                 self.current_speaker_id.pop(room_id, None)
                
#             print(f"Shutdown complete for room {room_id}")
#         else:
#             # Stop all threads across all rooms
#             for room_id in list(self.threads.keys()):
#                 self.stop_threads(room_id)

#     # Helper methods
#     def get_room_user_id(self, room_id, user_id):
#         return f"{room_id}_{user_id}"

#     def broadcast_results(self, room_id, lang_code, text, audio, recipient_ids):
#         """Broadcast translation results to recipients."""
#         for recipient_id in recipient_ids:
#             target_socket_id = user_sockets.get(recipient_id)
#             try:
#                 # Only send audio to recipients who aren't the speaker
#                 if recipient_id != self.current_speaker_id.get(room_id, "unknown"):
#                     # Broadcast audio
#                     audio_np = np.array(audio, dtype=np.float32)
#                     audio_file = f"{room_id}_{lang_code}_translated_audio.wav"
#                     sf.write(audio_file, audio_np, 16000)
                    
#                     with open(audio_file, "rb") as f:
#                         self.socketio.emit(
#                             "receive_audio",
#                             {"audio": f.read()},
#                             room=target_socket_id
#                         )
                
#                 # Broadcast text to all recipients
#                 print(f"Broadcasting {lang_code}: {text}")
#                 self.socketio.emit(
#                     "transcription",
#                     {
#                         "english": "",
#                         "translated": text,
#                         "sender_user_id": self.current_speaker_id.get(room_id)
#                     },
#                     to=target_socket_id,
#                 )
#             except Exception as e:
#                 print(f"Broadcast error: {e}")

#     def update_language_mappings(self, room_id):
#         """Update language mappings for a room."""
#         if room_id not in self.language_mapping:
#             self.language_mapping[room_id] = {}
        
#         # Initialize mapping with all available languages
#         self.language_mapping[room_id] = {lang_code: [] for lang_code in list(languages.values())}
        
#         # Get all users in the room
#         room_users = {uid: details for uid, details in user_session_details.items() 
#                      if details.get("room_id") == room_id}
        
#         # Update mappings based on user language preferences
#         for recipient_id, details in room_users.items():
#             tgt_lang = details.get("language", "Unknown")
#             if tgt_lang in languages:
#                 tgt_lang_code = languages[tgt_lang]
#                 self.language_mapping[room_id][tgt_lang_code].append(recipient_id)
#             else:
#                 print(f"Warning: Unknown language '{tgt_lang}' for user {recipient_id}")

#     def start_threads(self, room_id, start_up=False):
#         """Start processing threads for a room."""
#         print(f"Starting threads for room {room_id}, start_up={start_up}")
        
#         # Reset flags
#         self.stop_thread_flags[room_id] = False
#         self.shutdown_in_progress[room_id] = False
        
#         # Initialize current speaker ID if not set
#         if room_id not in self.current_speaker_id:
#             # Set a default speaker ID or get it from somewhere
#             self.current_speaker_id[room_id] = "default_speaker"
#             print(f"Set default speaker ID for room {room_id}")

#         if start_up:
#             # Update language mappings based on current users in the room
#             self.update_language_mappings(room_id)
            
#             # Initialize room-specific collections
#             if room_id not in self.threads:
#                 self.threads[room_id] = {}
            
#             # Initialize buffers and locks
#             self.audio_buffer[room_id] = np.array([], np.float32)
#             self.audio_buffer_locks[room_id] = threading.Lock()
            
#             self.transcription_buffer[room_id] = []
#             self.transcription_buffer_locks[room_id] = threading.Lock()
            
#             self.translation_buffers[room_id] = {}
#             self.translation_buffer_locks[room_id] = {}
            
#             # Initialize WebSocket connections
#             self.transcription_ws[room_id] = None
#             self.transcription_ws_locks[room_id] = threading.Lock()
            
#             self.translation_ws[room_id] = {}
#             self.translation_ws_locks[room_id] = {}
            
#             # Initialize translation-specific structures for each language
#             for lang_code in self.language_mapping[room_id]:
#                 if lang_code != "eng" and len(self.language_mapping[room_id][lang_code]) > 0:
#                     self.translation_buffers[room_id][lang_code] = []
#                     self.translation_buffer_locks[room_id][lang_code] = threading.Lock()
#                     self.translation_ws[room_id][lang_code] = None
#                     self.translation_ws_locks[room_id][lang_code] = threading.Lock()
            
#             # Start transcription thread (single thread for English)
#             transcription_thread = threading.Thread(
#                 target=self.transcription_thread_wrapper,
#                 args=(room_id,)
#             )
#             transcription_thread.daemon = True
#             self.threads[room_id]["transcription"] = transcription_thread
#             print(f"Starting transcription thread for room {room_id}")
#             transcription_thread.start()
            
#             # Start translation threads (one per target language)
#             for lang_code in self.language_mapping[room_id]:
#                 # Skip English as it doesn't need translation
#                 if lang_code != "eng" and len(self.language_mapping[room_id][lang_code]) > 0:
#                     translation_thread = threading.Thread(
#                         target=self.translation_thread_wrapper,
#                         args=(room_id, lang_code)
#                     )
#                     translation_thread.daemon = True
#                     self.threads[room_id][f"translation_{lang_code}"] = translation_thread
#                     print(f"Starting translation thread for room {room_id}, language {lang_code}")
#                     translation_thread.start()


# local translation.
class AudioHandler:
    def __init__(self):
        # Room-specific data structures
        self.threads = {}  # {room_id: {thread_type: thread}}
        self.audio_buffer = {}  # {room_id: buffer} - Shared audio buffer for each room
        self.audio_buffer_locks = {}  # {room_id: lock}
        
        # Transcription results buffer
        self.transcription_buffer = {}  # {room_id: {text, audio}} - Intermediate buffer for English transcriptions
        self.transcription_buffer_locks = {}  # {room_id: lock}
        
        # Language-specific translation buffers
        self.translation_buffers = {}  # {room_id: {lang_code: [{text, audio}]}}
        self.translation_buffer_locks = {}  # {room_id: {lang_code: lock}}
        
        self.language_mapping = {}  # {room_id: {lang_code: [user_ids]}}
        
        self.stop_thread_flags = {}  # {room_id: bool}
        self.processing_complete = {}  # {room_id: {thread_type: Event}}
        self.batch_size = AudioConfig.SAMPLERATE * 1
        self.socketio = None
        
        self.current_speaker_id = {}
        self.user_counts = {}  # {room_id: count}
        self.shutdown_in_progress = {}  # {room_id: bool}
        
        # WebSocket connections
        self.transcription_ws = {}  # {room_id: websocket} - Single websocket for English transcription
        self.transcription_ws_locks = {}  # {room_id: lock}
        
        # We don't need translation WebSockets anymore since we'll use local translation
        # But we'll keep the empty dictionaries for compatibility
        self.translation_ws = {}  # {room_id: {lang_code: websocket}}
        self.translation_ws_locks = {}  # {room_id: {lang_code: lock}}
        
        # Event loop for asyncio
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.loop_thread.start()
    
    def _run_event_loop(self):
        """Run the asyncio event loop in a separate thread."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def handle_audio(self, data, socketio, room_id):
        """Handle incoming audio data for a specific room by adding to shared buffer."""
        if not self.socketio:
            self.socketio = socketio

        # Don't accept new audio if shutdown is in progress
        if self.shutdown_in_progress.get(room_id, False):
            return

        # Start threads if not already running for this room
        if room_id not in self.threads:
            print(f"Starting new threads for room {room_id}")
            self.start_threads(room_id, start_up=True)
        
        if room_id in self.current_speaker_id:
            segment_handler.add_segment(room_id=room_id,
                                        audio=data,
                                        speaker=self.current_speaker_id[room_id])

            # Add audio data to shared buffer for the room
            self.add_audio_to_shared_buffer(room_id, data)
        else:
            print(f"Warning: No current speaker ID set for room {room_id}")

    def add_audio_to_shared_buffer(self, room_id, audio):
        """Add audio data to the shared buffer for processing."""
        if room_id not in self.audio_buffer_locks:
            self.audio_buffer_locks[room_id] = threading.Lock()
            self.audio_buffer[room_id] = np.array([], np.float32)
            
        tensor_samples = torch.tensor(audio, dtype=torch.float32)
        resampled_audio = torchaudio.functional.resample(tensor_samples, orig_freq=48000, new_freq=16_000).cpu().numpy()
        
        with self.audio_buffer_locks[room_id]:
            self.audio_buffer[room_id] = np.append(self.audio_buffer[room_id], resampled_audio)

    async def establish_transcription_websocket(self, room_id):
        """Establish WebSocket connection for English transcription."""
        try:
            print(f"Establishing transcription WebSocket connection for room {room_id}")
            ws = await websockets.connect(ServerConfig.ws_server_url, max_size=1024*1024*10)
            
            # Initialize WebSocket connection
            room_user_id = self.get_room_user_id(room_id, "eng")
            
            # Send initial message to reset buffers
            payload = {
                "client_id": room_user_id,
                "audio_data": [],
                "sampling_rate": AudioConfig.SAMPLERATE,
                "tgt_lang": "eng",  # Always English for transcription
                "reset_buffers": "true"
            }
            await ws.send(json.dumps(payload))
            print(f"Initial reset message sent for transcription {room_user_id}")
            
            # Wait for initial response
            response = await ws.recv()
            print(f"Transcription WebSocket connection established for room {room_id}")
            
            return ws
        except Exception as e:
            print(f"Error establishing transcription WebSocket connection for room {room_id}: {e}")
            return None

    # Local translation function
    def local_translate(self, text: str, lang_code: str) -> str:
        """Perform local translation by prefixing text with language code."""
        tgt_lang = seamless_nllb_lang_mapping.get(lang_code, nllb_languages.English)
        translated_text = nllb.translate_(text= text, src_lang =  nllb_languages.English,
                                     tgt_lang = tgt_lang )
        return translated_text

    async def send_audio_for_transcription(self, ws, room_user_id, audio_samples, reset_buffers="false"):
        """Send audio data via WebSocket for English transcription and receive response."""
        if ws is None:
            print(f"Cannot send audio: Transcription WebSocket connection is None for {room_user_id}")
            return None
            
        try:
            payload = {
                "client_id": room_user_id,
                "audio_data": audio_samples.tolist(),
                "sampling_rate": AudioConfig.SAMPLERATE,
                "tgt_lang": "eng",  # Always English for transcription
                "reset_buffers": reset_buffers
            }
            
            await ws.send(json.dumps(payload))
            response = await ws.recv()
            return json.loads(response)
        except Exception as e:
            print(f"Error sending audio for transcription via WebSocket for {room_user_id}: {e}")
            return None

    def add_to_transcription_buffer(self, room_id, text, audio):
        """Add transcription result to the buffer."""
        if room_id not in self.transcription_buffer_locks:
            self.transcription_buffer_locks[room_id] = threading.Lock()
            self.transcription_buffer[room_id] = []
            
        with self.transcription_buffer_locks[room_id]:
            self.transcription_buffer[room_id].append({"text": text, "audio": audio})

    def add_to_translation_buffer(self, room_id, lang_code, text, audio):
        """Add translation result to the language-specific buffer."""
        if room_id not in self.translation_buffer_locks:
            self.translation_buffer_locks[room_id] = {}
            self.translation_buffers[room_id] = {}
            
        if lang_code not in self.translation_buffer_locks[room_id]:
            self.translation_buffer_locks[room_id][lang_code] = threading.Lock()
            self.translation_buffers[room_id][lang_code] = []
            
        with self.translation_buffer_locks[room_id][lang_code]:
            self.translation_buffers[room_id][lang_code].append({"text": text, "audio": audio})

    async def transcription_thread_async(self, room_id):
        """Process audio for transcription to English."""
        print(f"Starting transcription thread for room {room_id}")
        
        # Check if we should stop before even starting
        if self.stop_thread_flags.get(room_id, False):
            print(f"Transcription thread for {room_id} asked to stop before starting")
            return
        
        try:
            # Establish WebSocket connection
            ws = await self.establish_transcription_websocket(room_id)
            if not ws:
                print(f"Failed to establish transcription WebSocket connection for room {room_id}")
                return
            
            # Store the websocket connection
            with self.transcription_ws_locks[room_id]:
                self.transcription_ws[room_id] = ws
            
            while not self.stop_thread_flags.get(room_id, False):
                # Check if the connection is still valid
                if ws.close_code is not None:
                    print(f"Transcription WebSocket closed unexpectedly for {room_id}, attempting to reconnect")
                    ws = await self.establish_transcription_websocket(room_id)
                    if not ws:
                        print(f"Failed to re-establish transcription WebSocket connection for {room_id}")
                        break
                    
                    with self.transcription_ws_locks[room_id]:
                        self.transcription_ws[room_id] = ws
                
                samples = np.array([], np.float32)
                
                # Process audio samples if available
                with self.audio_buffer_locks[room_id]:
                    buffer = self.audio_buffer[room_id]
                    if len(buffer) > 0:
                        # Process in smaller chunks to ensure smooth handling
                        chunk_size = min(self.batch_size, len(buffer))
                        samples = buffer[:chunk_size]
                        self.audio_buffer[room_id] = buffer[chunk_size:]
                
                if len(samples) > 0:
                    room_user_id = self.get_room_user_id(room_id, "eng")
                    
                    # Process the audio chunk using WebSocket
                    with self.transcription_ws_locks[room_id]:
                        response = await self.send_audio_for_transcription(
                            self.transcription_ws[room_id],
                            room_user_id,
                            samples
                        )
                        
                        if response:
                            transcriptions = response.get('transcriptions', "")
                            if transcriptions:
                                try:
                                    text, audio = eval(transcriptions)
                                    print(f"Transcription: {text}")
                                    if text[0] != "[" and get_unique_words_percentage(text) > 0.4:
                                        # Add to transcription buffer for distribution
                                        self.add_to_transcription_buffer(room_id, text, audio)

                                        # For English users, directly deliver the transcription
                                        if "eng" in self.language_mapping[room_id] and self.language_mapping[room_id]["eng"]:
                                            self.broadcast_results(room_id, "eng", text, audio, self.language_mapping[room_id]["eng"])
                                except Exception as e:
                                    print(f"Error processing transcription response: {e}")
                else:
                    # If no audio to process, sleep briefly to avoid busy-waiting
                    await asyncio.sleep(0.1)
            
            # Clean up WebSocket connection
            print(f"Closing transcription WebSocket for {room_id}")
            if ws.close_code is None:
                await ws.close()
            
            # Signal completion before exiting
            if room_id in self.processing_complete and "transcription" in self.processing_complete[room_id]:
                print(f"Setting completion event for transcription thread in room {room_id}")
                self.processing_complete[room_id]["transcription"].set()
                
        except Exception as e:
            print(f"Exception in transcription_thread_async for room {room_id}: {e}")
            traceback.print_exc()
            if room_id in self.processing_complete and "transcription" in self.processing_complete[room_id]:
                self.processing_complete[room_id]["transcription"].set()

    async def translation_thread_async(self, room_id, lang_code):
        """Process transcription results for translation to a specific language using local translation."""
        # Skip translation for English as it's already transcribed
        if lang_code == "eng":
            print(f"Skipping translation thread for English in room {room_id}")
            return
            
        print(f"Starting translation thread for room {room_id}, language {lang_code}")
        
        # Check if we should stop before even starting
        if self.stop_thread_flags.get(room_id, False):
            print(f"Translation thread for {room_id}/{lang_code} asked to stop before starting")
            return
        
        try:
            while not self.stop_thread_flags.get(room_id, False):
                # Check for transcription results to translate
                items_to_process = []
                with self.transcription_buffer_locks[room_id]:
                    if self.transcription_buffer[room_id]:
                        items_to_process = self.transcription_buffer[room_id].copy()
                        self.transcription_buffer[room_id] = []
                
                for item in items_to_process:
                    # Use local translation instead of external service
                    translated_text = self.local_translate(item["text"], lang_code)
                    translated_audio = item["audio"]  # Use original audio
                    
                    print(f"Translation ({lang_code}): {translated_text}")
                    
                    # Broadcast to users of this language
                    if lang_code in self.language_mapping[room_id] and self.language_mapping[room_id][lang_code]:
                        self.broadcast_results(
                            room_id, 
                            lang_code, 
                            translated_text, 
                            translated_audio,  # Use original audio
                            self.language_mapping[room_id][lang_code]
                        )
                
                if not items_to_process:
                    # If no transcriptions to process, sleep briefly to avoid busy-waiting
                    await asyncio.sleep(0.1)
            
            # Signal completion before exiting
            if room_id in self.processing_complete and f"translation_{lang_code}" in self.processing_complete[room_id]:
                print(f"Setting completion event for translation thread {lang_code} in room {room_id}")
                self.processing_complete[room_id][f"translation_{lang_code}"].set()
                
        except Exception as e:
            print(f"Exception in translation_thread_async for room {room_id}, language {lang_code}: {e}")
            traceback.print_exc()
            if room_id in self.processing_complete and f"translation_{lang_code}" in self.processing_complete[room_id]:
                self.processing_complete[room_id][f"translation_{lang_code}"].set()

    def transcription_thread_wrapper(self, room_id):
        """Wrapper to run async transcription thread in a sync context."""
        future = asyncio.run_coroutine_threadsafe(
            self.transcription_thread_async(room_id),
            self.loop
        )
        
        # Handle any exceptions from the future
        try:
            future.result()
        except Exception as e:
            print(f"Error in transcription_thread_wrapper: {e}")
            traceback.print_exc()

    def translation_thread_wrapper(self, room_id, lang_code):
        """Wrapper to run async translation thread in a sync context."""
        future = asyncio.run_coroutine_threadsafe(
            self.translation_thread_async(room_id, lang_code),
            self.loop
        )
        
        # Handle any exceptions from the future
        try:
            future.result()
        except Exception as e:
            print(f"Error in translation_thread_wrapper: {e}")
            traceback.print_exc()

    async def stop_websocket_connections(self, room_id):
        """Close all WebSocket connections for a room."""
        # Close transcription WebSocket
        if room_id in self.transcription_ws:
            try:
                ws = self.transcription_ws[room_id]
                if ws and ws.close_code is None:
                    # Send final cleanup message
                    room_user_id = self.get_room_user_id(room_id, "eng")
                    await self.send_audio_for_transcription(
                        ws, 
                        room_user_id, 
                        np.array([], np.float32), 
                        reset_buffers="true"
                    )
                    await ws.close()
            except Exception as e:
                print(f"Error closing transcription WebSocket for {room_id}: {e}")
        
        # We don't need to close translation WebSockets anymore since we're using local translation
        # but we'll keep the code structure intact for potential future changes

    def stop_threads(self, room_id=None):
        """Stop threads gracefully, ensuring all buffered audio is processed."""
        if room_id:
            print(f"Initiating shutdown for room {room_id}")
            
            # First check if the room exists in our data structures
            if room_id not in self.threads:
                print(f"No threads found for room {room_id}, skipping shutdown")
                return
                
            # Log audio buffer status
            if room_id in self.audio_buffer:
                print(f"Room {room_id} has {len(self.audio_buffer[room_id])/16000} seconds of unprocessed audio during shut down")
            
            # Mark shutdown in progress to prevent new audio from being added
            self.shutdown_in_progress[room_id] = True
            
            # Set stop flag for the room to signal threads to terminate
            self.stop_thread_flags[room_id] = True          
            
            # Initialize completion events for each thread
            self.processing_complete[room_id] = {
                "transcription": threading.Event()
            }
            
            # Add completion events for translation threads
            for lang_code in self.language_mapping.get(room_id, {}):
                if lang_code != "eng":  # English doesn't have a translation thread
                    self.processing_complete[room_id][f"translation_{lang_code}"] = threading.Event()
            
            # Wait for all threads to finish processing remaining audio
            print(f"Waiting for processing completion in room {room_id}")
            for thread_type, event in self.processing_complete[room_id].items():
                # Longer timeout to ensure processing completes
                if not event.wait(timeout=10.0):
                    print(f"Warning: Timeout waiting for {thread_type} processing in room {room_id}")
            print(f"All processes have completed. Initializing shutdown.")  
            
            # Close all WebSocket connections
            future = asyncio.run_coroutine_threadsafe(
                self.stop_websocket_connections(room_id),
                self.loop
            )
            
            # Wait for the future to complete with a timeout
            try:
                future.result(timeout=5.0)
            except Exception as e:
                print(f"Error stopping WebSocket connections: {e}")
            
            # Check remaining audio after processing
            if room_id in self.audio_buffer:
                print(f"Room {room_id} has {len(self.audio_buffer[room_id])/16000} seconds of unprocessed audio after shut down")
            
            # Clean up room resources
            print(f"Cleaning up resources for room {room_id}")
            
            # Clean up room-specific data structures
            if room_id in self.threads:
                # Stop all threads first
                for thread in self.threads[room_id].values():
                    if thread.is_alive():
                        thread.join(timeout=2.0)
                
                # Now clean up the data structures
                self.threads.pop(room_id, None)
                self.audio_buffer.pop(room_id, None)
                self.audio_buffer_locks.pop(room_id, None)
                self.transcription_buffer.pop(room_id, None)
                self.transcription_buffer_locks.pop(room_id, None)
                self.translation_buffers.pop(room_id, None)
                self.translation_buffer_locks.pop(room_id, None)
                self.language_mapping.pop(room_id, None)
                self.processing_complete.pop(room_id, None)
                self.shutdown_in_progress.pop(room_id, None)
                self.transcription_ws.pop(room_id, None)
                self.transcription_ws_locks.pop(room_id, None)
                self.translation_ws.pop(room_id, None)
                self.translation_ws_locks.pop(room_id, None)
                self.stop_thread_flags.pop(room_id, None)
                self.current_speaker_id.pop(room_id, None)
                
            print(f"Shutdown complete for room {room_id}")
        else:
            # Stop all threads across all rooms
            for room_id in list(self.threads.keys()):
                self.stop_threads(room_id)

    # Helper methods
    def get_room_user_id(self, room_id, user_id):
        return f"{room_id}_{user_id}"

    def broadcast_results(self, room_id, lang_code, text, audio, recipient_ids):
        """Broadcast translation results to recipients."""
        for recipient_id in recipient_ids:
            target_socket_id = user_sockets.get(recipient_id)
            try:
                # Only send audio to recipients who aren't the speaker
                if recipient_id != self.current_speaker_id.get(room_id, "unknown"):
                    # Broadcast audio
                    audio_np = np.array(audio, dtype=np.float32)
                    audio_file = f"{room_id}_{lang_code}_translated_audio.wav"
                    sf.write(audio_file, audio_np, 16000)
                    
                    with open(audio_file, "rb") as f:
                        self.socketio.emit(
                            "receive_audio",
                            {"audio": f.read()},
                            room=target_socket_id
                        )
                
                # Broadcast text to all recipients
                print(f"Broadcasting {lang_code}: {text}")
                self.socketio.emit(
                    "transcription",
                    {
                        "english": "",
                        "translated": text,
                        "sender_user_id": self.current_speaker_id.get(room_id)
                    },
                    to=target_socket_id,
                )
            except Exception as e:
                print(f"Broadcast error: {e}")

    def update_language_mappings(self, room_id):
        """Update language mappings for a room."""
        if room_id not in self.language_mapping:
            self.language_mapping[room_id] = {}
        
        # Initialize mapping with all available languages
        self.language_mapping[room_id] = {lang_code: [] for lang_code in list(languages.values())}
        
        # Get all users in the room
        room_users = {uid: details for uid, details in user_session_details.items() 
                     if details.get("room_id") == room_id}
        
        # Update mappings based on user language preferences
        for recipient_id, details in room_users.items():
            tgt_lang = details.get("language", "Unknown")
            if tgt_lang in languages:
                tgt_lang_code = languages[tgt_lang]
                self.language_mapping[room_id][tgt_lang_code].append(recipient_id)
            else:
                print(f"Warning: Unknown language '{tgt_lang}' for user {recipient_id}")

    def start_threads(self, room_id, start_up=False):
        """Start processing threads for a room."""
        print(f"Starting threads for room {room_id}, start_up={start_up}")
        
        # Reset flags
        self.stop_thread_flags[room_id] = False
        self.shutdown_in_progress[room_id] = False
        
        # Initialize current speaker ID if not set
        if room_id not in self.current_speaker_id:
            # Set a default speaker ID or get it from somewhere
            self.current_speaker_id[room_id] = "default_speaker"
            print(f"Set default speaker ID for room {room_id}")

        if start_up:
            # Update language mappings based on current users in the room
            self.update_language_mappings(room_id)
            
            # Initialize room-specific collections
            if room_id not in self.threads:
                self.threads[room_id] = {}
            
            # Initialize buffers and locks
            self.audio_buffer[room_id] = np.array([], np.float32)
            self.audio_buffer_locks[room_id] = threading.Lock()
            
            self.transcription_buffer[room_id] = []
            self.transcription_buffer_locks[room_id] = threading.Lock()
            
            self.translation_buffers[room_id] = {}
            self.translation_buffer_locks[room_id] = {}
            
            # Initialize WebSocket connections - we only need transcription
            self.transcription_ws[room_id] = None
            self.transcription_ws_locks[room_id] = threading.Lock()
            
            # We'll keep empty translation_ws dictionaries for compatibility
            self.translation_ws[room_id] = {}
            self.translation_ws_locks[room_id] = {}
            
            # Initialize translation-specific structures for each language
            for lang_code in self.language_mapping[room_id]:
                if lang_code != "eng" and len(self.language_mapping[room_id][lang_code]) > 0:
                    self.translation_buffers[room_id][lang_code] = []
                    self.translation_buffer_locks[room_id][lang_code] = threading.Lock()
                    self.translation_ws[room_id][lang_code] = None  # Not actually used anymore
                    self.translation_ws_locks[room_id][lang_code] = threading.Lock()
            
            # Start transcription thread (single thread for English)
            transcription_thread = threading.Thread(
                target=self.transcription_thread_wrapper,
                args=(room_id,)
            )
            transcription_thread.daemon = True
            self.threads[room_id]["transcription"] = transcription_thread
            print(f"Starting transcription thread for room {room_id}")
            transcription_thread.start()
            
            # Start translation threads (one per target language)
            for lang_code in self.language_mapping[room_id]:
                # Skip English as it doesn't need translation
                if lang_code != "eng" and len(self.language_mapping[room_id][lang_code]) > 0:
                    translation_thread = threading.Thread(
                        target=self.translation_thread_wrapper,
                        args=(room_id, lang_code)
                    )
                    translation_thread.daemon = True
                    self.threads[room_id][f"translation_{lang_code}"] = translation_thread
                    print(f"Starting translation thread for room {room_id}, language {lang_code}")
                    translation_thread.start()

# Create singleton instance
handler = AudioHandler()

# Start asyncio event loop in a separate thread to handle WebSocket operations
def start_asyncio_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

asyncio_thread = threading.Thread(target=start_asyncio_loop, args=(handler.loop,), daemon=True)
asyncio_thread.start()

def send_server_request(user_id,  audio, tgt_lang, reset_buffers = "false"):
    try:

        if  reset_buffers == "true":
            response = requests.post(f"http://{ServerConfig.server_host}:{ServerConfig.server_port}/cleanup/{user_id}")
            return response


        else:
            payload = {
                "client_id": user_id,
                "audio_data": audio.tolist(),
                "sampling_rate": AudioConfig.SAMPLERATE,
                "tgt_lang": tgt_lang,
                "reset_buffers": reset_buffers
            }
            
            response = requests.post(
                ServerConfig.server_url,
                data=json.dumps(payload),
                headers=ServerConfig.headers
            )
            
            
            if response.status_code == 200:
                # print(f"server responded with {response.text}")
                return response
            
        print(f"Request failed with status code: {response.status_code}")
    except Exception as e:
        print(f"Server request error: {e}")
    return None

def handle_response(response):
    if response is not None:
        try:
            transcriptions = eval(response.text).get('transcriptions', "")
            # print(f"transcripts:{transcriptions} of type {type(transcriptions)}")
                                    
            if transcriptions:
                text = transcriptions

                if text[0] != "[":
                    print(f"Transcribed Text: {text}")
                    # all_transcriptions.append(text)
                return text

        except Exception as e:
            print(f"Failed to decode response JSON: {e}")

    return None

def register_socket_handlers(socketio):
    @socketio.on("checkRoom")
    def check_room(data):
        room_id = data["roomId"]
        role = data.get("role")

        # Check if the room exists in the `rooms` dictionary
        room_exists = room_id in rooms

        # Allow instructors to create new rooms
        if not room_exists and role == "instructor":
            rooms[room_id] = []  # Initialize the room
            room_exists = True

        # Respond back to the frontend
        return {"exists": room_exists}

    @socketio.on("join")
    def on_join(data):
        room = data["room"]
        user_details = data["userDetails"]
        user_role = user_details.get("role")
        
         # Check if the room exists, and handle based on the user's role
        if room not in rooms:
            if user_role != "instructor":
                socketio.emit("join-error", {"error": "Room not created by an instructor."}, to=request.sid)
                return
            else:
                rooms[room] = []  # Instructors can create the room


        print("00000000000000000000 : ", user_details)

        # user_id = request.sid  # Unique Socket ID for the user
        user_id = user_details["user_id"]
        user_details['room_id'] = room
        user_sockets[user_id] = request.sid
        user_session_details[user_id] = user_details

        # Add user to room's user list
        # if room not in rooms:
        #     rooms[room] = []

        # Check if the user is already in the room
        if user_id not in rooms[room]:
            rooms[room].append(user_id)
        # join_room(room)
        # emit('room_update', {'users': rooms[room]}, room=room)
        # rooms[room].append(user_id)

        join_room(room)
        send(f"User {user_id} has entered the room {room}", to=room)

        # Print number of users in the room
        print(f"Room {room} has {len(rooms[room])} users: {rooms[room]}")

        # Prepare user details for all users in the room
        room_user_details = [user_session_details[uid] for uid in rooms[room]]

        print("all user details : ", room_user_details)
        # Send the updated list of users and their details to everyone in the room
        socketio.emit(
            "updateUsers",
            {"users": rooms[room], "userDetails": room_user_details},
            room=room,
        )
        print("room id and request id :", room, request.sid)
        socketio.emit('new-participant', {
            'socketId': request.sid,
        }, room=room)
        print("Emitting 'new-participant' with socket ID:", request.sid)
          

    @socketio.on('offer')
    def handle_offer(data):
        print("text2---------", data)
        room = data.get('room')
        offer = data.get('offer')
        target = data.get('target')

        print("text3------", room, offer, target)
        if not room or not offer or not target:
            print(f"Missing room ID {room}, offer {offer}, or target {target}")
            return "Missing room ID, offer, or target", 400
        
        print(f"Received offer for room {room} from {request.sid} to {target}")
        socketio.emit('offer', {
            'offer': offer,
            'socketId': request.sid
        }, room=target)

    # Handle WebRTC answer
    @socketio.on('answer')
    def handle_answer(data):
        print("text3-------------------------", data)
        room = data.get('room')
        answer = data.get('answer')
        target = data.get('target')

        print("text-4----------------", room,answer,target)
        if not room or not answer or not target:
            print(f"Missing room ID {room}, answer {answer}, or target {target}")
            return "Missing room ID, answer, or target", 400

        print(f"Received answer for room {room} from {request.sid} to {target}")
        socketio.emit('answer', {
            'answer': answer,
            'socketId': request.sid
        }, room=target)

    # Handle ICE candidates
    @socketio.on('ice-candidate')
    def handle_ice_candidate(data):
        print("entering into ICE CCCCCCCCCC", data)
        room = data.get('room')
        candidate = data.get('candidate')
        target = data.get('target')
        user_id = data.get('userId')
        print("printing test 1 -----------------------", room, candidate,target, user_id)
        if not room or not candidate or not target:
            print(f"Missing room ID {room}, candidate {candidate}, or target {target}")
            return "Missing room ID, ICE candidate, or target", 400

        print(f"Received ICE candidate for room {room} from {request.sid} to {target}")
        socketio.emit('ice-candidate', {
            'candidate': candidate,
            'socketId': request.sid
        }, room=target)

# ==============================================================================
    @socketio.on("screen-sharing-status")
    def screen_sharing_status(data):
        print('==========ssssssssssssssssssssssss=============')
        print(data)
        print('==========sssssssssssssssssssssssss=============')
        stop_screen=False
        if data == False:
            stop_screen=True
        
        # Respond back to the frontend
        socketio.emit('screenshare', stop_screen)

    @socketio.on("stop_screen")
    def stop_screen(data):
        print('==========eeeeeeeeeeeeeeeeeeeeee=============')
        print(data)
        print('==========eeeeeeeeeeeeeeeeeeeeee=============')
        stop_screen=False
        if data == False:
            stop_screen=True
        
        # Respond back to the frontend
        socketio.emit('stopscreen', stop_screen)
# ================================================================================================

#------------------------------------------------------------------------------------------    

        # Send the updated list of users to everyone in the room
        # socketio.emit("updateUsers", rooms[room], room=room)

    @socketio.on("leave")
    def on_leave(data):
        room = data["room"]
        user_id = data["user_id"]  # Unique Socket ID for the user
        print(rooms,"..............the rooms_data...............")
        # Remove user from the room's user list
        if room in rooms and user_id in rooms[room]:
            rooms[room].remove(user_id)
            print(rooms,"..............POP_DATA_REMOVED...............")

        leave_room(room)
        send(f"User {user_id} has left the room {room}", to=room)

        # Print number of users remaining in the room
        print(f"Room {room} has {len(rooms[room])} users: {rooms[room]}")
        # Remove the user from user_sockets when they leave
        user_sockets.pop(user_id, None)

        # Prepare user details for all users in the room
        room_user_details = [user_session_details[uid] for uid in rooms[room]]

        print("all user details : ", room_user_details)
        # Send the updated list of users and their details to everyone in the room
        socketio.emit(
            "updateUsers",
            {"users": rooms[room], "userDetails": room_user_details},
            room=room,
        )

        # Send the updated list of users to everyone in the room
        # socketio.emit("updateUsers", rooms[room], room=room)
# ---------------added for ON and OFF mic ---------------------------------------
    @socketio.on("toggle_mic")
    def toggle_mic(data):
        socketio.emit('update_mic',data)
# ---------------------------------------------------------------------------------

    @socketio.on("close_room")
    def on_close_room(data):
        room = data["room"]

        print(rooms,"..............the rooms_data...............")
        if room in rooms:
            send(f"The room {room} is now closed", to=room)
            close_room(room)
            # Clear the room's user list
            del rooms[room]
            print(rooms,"..............after pop....................")
        print(f"Room {room} has been closed.")

        socketio.emit("roomClosed",  to=room)

    @socketio.on("end_call")
    def on_end_call(data):
        global handler
        room = data["room"]
        print(room,"...............end_call..........")
        instructor_id = data.get("user_id")  # Instructor's user ID
       
        if handler.stop_thread_flags.get(room)!= False:
            handler.stop_threads(room_id =  room)
        
        segment_handler.dump_segments(room_id = room)
        
        if room in rooms:
            # Notify everyone in the room that the call is ending
            socketio.emit("roomClosed", {"message": "The call has been ended by the instructor."}, to=room)

            # Iterate through users in the room
            user_ids = rooms[room]
            for user_id in user_ids:
                user_socket = user_sockets.get(user_id)  # Get the user's socket ID
                if user_socket:
                    # Notify the individual user
                    socketio.emit(
                        "userDisconnected",
                        {"message": "The call has been ended by the instructor."},
                        room=user_socket
                    )
                    # Remove the user from the room
                    leave_room(room, sid=user_socket)

            # Clear chat history for the room
            chat_history.pop(room, None)
            print(rooms,"..............the rooms_data...............")
            # Clear the room data
            rooms.pop(room, None)
            print(rooms,"..............after pop....................")
            for user_id in user_ids:
                user_sockets.pop(user_id, None)
                user_session_details.pop(user_id, None)

            print(f"Room {room} ended by instructor {instructor_id}. All users disconnected.")
        else:
            print(f"Room {room} does not exist or has already been closed.")
    
    

    def convert_file_to_base64(file, file_name):
        """Convert binary file to Base64 with its MIME type."""
        base64_file = base64.b64encode(file).decode('utf-8')
        # Guess MIME type based on the file name or default to application/octet-stream
        mime_type = mimetypes.guess_type(file_name)[0] or "application/octet-stream"
        return f"data:{mime_type};base64,{base64_file}"
# =============================================================================================================================
    @socketio.on("message")
    def handle_message(data):
        print("Enter into message ------------------------------------------------yes")
        room = data["room"]
        message = data["message"]
        sender_user_id = data["user_id"]
        name = data["name"]

        file = data.get("file")  # This is expected to be the binary content of the file
        file_name = data.get("file_name", "unknown_file")  # Get the file name with extension
        timestamp = data.get("timestamp") 

        print("namaename000000000000000000",name)

        current_user_details = user_session_details.get(sender_user_id, {})
        username = current_user_details.get("username", "Unknown User")
        src_language = current_user_details.get("language", "en")

        # Prepare message data to be sent
        message_data = {
            "sender_username": username,
            "sender_id": sender_user_id,
            "is_private": False,
            "timestamp": timestamp, 
        }
        sender_message_data = {
            "sender_username": username,
            "sender_id": sender_user_id,
            "is_private": False,
            "timestamp": timestamp, 
        }
        receiver_message_data = {
            "sender_username": username,
            "sender_id": sender_user_id,
            "is_private": False,
            "timestamp": timestamp, 
        }
        if file:
            file_extension = os.path.splitext(file_name)[1].lower()
            if file_extension == ".pdf":
               
                print(f"Processing PDF file: {file_name}")

                # Directory to store uploaded PDFs
                upload_dir = "uploaded_pdfs"
                # upload_dir = "uploaded_pdfs"output_doc
                os.makedirs(upload_dir, exist_ok=True)
                pdf_path = os.path.join(upload_dir, file_name)

                # Save the uploaded PDF
                with open(pdf_path, "wb") as pdf_file:
                    pdf_file.write(file)

                for user_id, details in user_session_details.items():
                    target_socket_id = user_sockets.get(user_id)
                    target_language = details.get("language", "en")

                    # If the current user is the sender, send the original PDF
                    if sender_user_id == user_id:
                        print(file,".............file.........")
                        print(file_name,".............file_name.........")
                        translated_file = convert_file_to_base64(file, file_name)
                        sender_message_data["file"] = translated_file
                        sender_message_data["file_name"] = file_name 
                        
                        
                        socketio.emit("message", sender_message_data, to=target_socket_id)
                    else:
                        # Process and translate the PDF for other users
                        translated_md_path = pdf_translation(pdf_path, src_language, target_language)
                        print(translated_md_path, ".............md_path...............")
                        if translated_md_path:
                            # Read the translated PDF file in binary mode
                            translated_pdf_path = translated_md_path.replace(".md", ".pdf")
                            if os.path.exists(translated_pdf_path):
                                with open(translated_pdf_path, "rb") as translated_pdf_file:
                                    translated_pdf_data = translated_pdf_file.read()

                                # Convert the translated PDF to base64
                                translated_file = convert_file_to_base64(translated_pdf_data, os.path.basename(translated_pdf_path))
                                receiver_message_data = {
                                    "file": translated_file,
                                    "file_name": os.path.basename(translated_pdf_path),
                                    "sender_username": username,
                                    "timestamp": timestamp, 
                                }
                                print("receiver_message_data",receiver_message_data)
                                # Send the translated PDF to the frontend
                                socketio.emit("message", receiver_message_data, to=target_socket_id)
                        
            else :
                # Convert the file to base64 with its MIME type
                translated_file = convert_file_to_base64(file, file_name)
                message_data["file"] = translated_file
                message_data["file_name"] = file_name 

                if message:
                    message_data["message"] = message
        elif message:
            message_data["message"] = message

        # Send the message with the necessary data
        send(message_data, to=room)

        # Translate and send the message to other users (excluding the sender)
        if message:
            for user_id, details in user_session_details.items():
                if sender_user_id == user_id:
                    continue

                target_socket_id = user_sockets.get(user_id)
                translated_text = text_to_text_translation(
                    message,
                    src_language,
                    details["language"],
                )

                print("translated text  : ", translated_text)
                # return translated_text
                socketio.emit(
                    "text-translation-completed", translated_text, to=target_socket_id
                )
                # =====added=
                socketio.emit(
                    "chat_history", translated_text, to=target_socket_id
                )
    @socketio.on("private_message")
    def handle_private_message(data):
        target_user = data["targetUser"]
        message = data["message"]
        user_id = data["user_id"]

        file = data.get("file")
        file_name = data.get("file_name", "unknown_file")
        timestamp = data.get("timestamp") 


        print("======================private_message===============================")
        # print("==============file_name========",file_name, "Received private message at:", timestamp)

        # Check if target user exists in the user_sockets mapping
        target_socket_id = user_sockets.get(target_user)
        sender_socket_id = user_sockets.get(user_id)
        target_user_details = user_session_details.get(target_user)
        current_user_details = user_session_details.get(user_id)
        target_username = target_user_details.get("username", "Unknown Target User")
        sender_username = current_user_details.get("username", "Unknown Sender")

        print("target_user_details : ", target_user_details, current_user_details)
        print(
            "target_user_details : ",
            target_user_details["language"],
            current_user_details["language"],
        )

        # Send private message to the target user only

        # Prepare the message payload
        private_message_data = {
            "message": message,
            "sender_username": sender_username,
            "receiver_username": target_username,
            "sender_id": user_id,
            "receiver_id": target_user,
            "is_private": True,
            "timestamp": timestamp,

        }
        sender_message_data = {
            "message": message,
            "sender_username": sender_username,
            "receiver_username": target_username,
            "sender_id": user_id,
            "receiver_id": target_user,
            "is_private": True,
            "timestamp": timestamp,

        }
        receiver_message_data = {
            "message": message,
            "sender_username": sender_username,
            "receiver_username": target_username,
            "sender_id": user_id,
            "receiver_id": target_user,
            "is_private": True,
            "timestamp": timestamp,

        }
        # Add the file if it exists
        if file:
            file_extension = os.path.splitext(file_name)[1].lower()
            # ------------------added-----
            if file_extension == ".pdf":
               
                print(f"Processing PDF file: {file_name}")

                # Directory to store uploaded PDFs
                upload_dir = "uploaded_pdfs"
                # upload_dir = "uploaded_pdfs"output_doc
                os.makedirs(upload_dir, exist_ok=True)
                pdf_path = os.path.join(upload_dir, file_name)

                # Save the uploaded PDF
                with open(pdf_path, "wb") as pdf_file:
                    pdf_file.write(file)
                src_language = current_user_details.get("language")
                target_language = target_user_details.get("language")
                if sender_socket_id:
                    translated_file = convert_file_to_base64(file, file_name)
                    # sender_message_data = {
                    #     "file": translated_file,
                    #     "file_name": file_name,
                    # }
                    sender_message_data["file"] = translated_file
                    sender_message_data["file_name"] = file_name 
                    
                    socketio.emit("message", sender_message_data, to=sender_socket_id)

                translated_md_path = pdf_translation(pdf_path, src_language, target_language)
                print(translated_md_path, ".............md_path...............")
                if translated_md_path:
                    # Read the translated PDF file in binary mode
                    translated_pdf_path = translated_md_path.replace(".md", ".pdf")
                    if os.path.exists(translated_pdf_path):
                        with open(translated_pdf_path, "rb") as translated_pdf_file:
                            translated_pdf_data = translated_pdf_file.read()

                        # Convert the translated PDF to base64
                        translated_file = convert_file_to_base64(translated_pdf_data, os.path.basename(translated_pdf_path))
                        receiver_message_data = {
                            "file": translated_file,
                            "file_name": os.path.basename(translated_pdf_path),
                            "sender_username": username,
                        }

                        # Send the translated PDF to the frontend
                        socketio.emit("message", receiver_message_data, to=target_socket_id)
                        # socketio.emit("message", private_message_data, to=sender_socket_id)

            # -----------added---------
            else:
                translated_file = convert_file_to_base64(file, file_name)  # Your existing utility function
                private_message_data["file"] = translated_file
                private_message_data["file_name"] = file_name

                print("Private message data being sent:", private_message_data)
        if target_socket_id:
            socketio.emit("message", private_message_data, to=target_socket_id)
            socketio.emit("message", private_message_data, to=sender_socket_id)

            # Translate the text part of the message if applicable
            if message:
                translated_text = text_to_text_translation(
                    message,
                    current_user_details.get("language"),
                    target_user_details.get("language"),
                )
                socketio.emit("text-translation-completed", translated_text, to=target_socket_id)
                socketio.emit("chat_history", translated_text, to=target_socket_id)

        else:
            # send(f"User {user_id}: {message}", to=data["room"])
            # Handle the case where the target user is not connected
            send(f"User {target_user} is not online", to=sender_socket_id)
   
    @socketio.on("translate-text")
    def handle_translate_text_to_local(data):
        """

        As of now for public messages we can't see the Translation, because below error.

        Need to Resolve this ERROR :
        The phrase "Already borrowed" indicates that the processor_m4t object is already in use in another thread,
          and you're trying to reuse it without releasing it first.

        """

        text = data["text"]
        sender_id = data["sender_id"]
        user_id = data["user_id"]
        # message_index = data["message_index"]

        sender_user_details = user_session_details.get(sender_id)
        current_user_details = user_session_details.get(user_id)

        # src_language_code = src_language[:3]

        # sender_language_code = sender_user_details["language"][:3]
        # current_user_language_code = current_user_details["language"][:3]
        sender_language_code = sender_user_details["language"]
        current_user_language_code = current_user_details["language"]

        print(
            "helooooooo master heart miss aayeyyyy : ",
            sender_language_code,
            current_user_language_code,
        )

        sender_socket_id = user_sockets.get(user_id)
        translated_text = text_to_text_translation(
            text, sender_language_code, current_user_language_code
        )

        print("translated text  : ", translated_text)
        # return translated_text
        socketio.emit(
            "text-translation-completed", translated_text, to=sender_socket_id
        )
        # =====added=
        socketio.emit(
            "chat_history", translated_text, to=sender_socket_id
        )
        # socketio.emit(
        #     "text-translation-completed", {"message_index": message_index, "translated_text": translated_text}, to=sender_socket_id
        # )
    
    @socketio.on("send_user_details")
    def handle_user_details(data):
        print("Received User Details:", data)
        global instructor_language  # Use global variable
        for user in data:
            if user["type"].lower() == "instructor":  # Check if user is Instructor
                lang_name = user["language"].lower()
                instructor_language = languages.get(lang_name, "unknown")
                # instructor_language = user["language"]
                print("Updated Instructor Language:", instructor_language)  # Debugging Output
    
    # room aware implementation.
    @socketio.on("audio")
    def handle_audio(data, room_data):
        global handler, old_len, full_audio, frames, now
        time_elapsed = (time.time() - now) if now is not None else 0
        print("time elapsed : ", time_elapsed)
        now = time.time()

        # Extract room data
        room = room_data["room"]
        user_id = room_data["user_id"]
        src_language = room_data["src_language"]
        sample_rate = room_data["sampleRate"]

        # Update current speaker
        if room not in handler.current_speaker_id or handler.current_speaker_id[room] != user_id:
            handler.current_speaker_id[room] = user_id
            print(handler.current_speaker_id)
            
        

        
        # Get users for the current room
        room_users = {}
        current_room_id = user_session_details.get('room_id')
        
        if current_room_id == room:
            # Get all user entries (excluding the room_id key)
            room_users = {k: v for k, v in user_session_details.items() 
                        if k != 'room_id' and isinstance(v, dict)}
        
        # Initialize room-specific user count if needed
        if room not in handler.user_counts:
            handler.user_counts[room] = 0
        
        # Check if user count changed for this room
        if handler.user_counts[room] != len(room_users):
            handler.user_counts[room] = len(room_users)
            handler.update_language_mappings(room)
            print("calling update mapping")
            print(f"Current room {room} users: {room_users}")
            print(f"Total user session details: {user_session_details}")

        # Process audio data
        if data:
            frames.append(data)
            audio_data = b"".join(frames)
            audio_array, sample_rate = torchaudio.load(io.BytesIO(audio_data))
            audio_array = audio_array.cpu().numpy().flatten()
            
            # Pass room context to handle_audio
            handler.handle_audio(audio_array[old_len:], socketio, room)
            old_len = len(audio_array)

    # Handle stop event to combine audio and save the file
    @socketio.on("stop")
    def handle_stop():
        global full_audio, frames
        print('stop audio is being called')
        if frames:
            unique_filename = str(uuid.uuid4())
            file_path = os.path.join(AUDIO_DIR, f"audio_{unique_filename}.wav")
            with wave.open(file_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(44100)
                wf.writeframes(b"".join(frames))
            
            
            handler.stop_threads()
            # frames.clear()



#---------------------------------------------------------------------------------------------------------

    @socketio.on("chat_history")
    def chat_history_data(data):
        room_id = data.get("roomId")
        instructor_name = data.get("instructorname")
        role = data.get("role")
        date_str = data.get("date")
        date_obj = datetime.strptime(date_str, "%d/%m/%Y")
        formatted_date = date_obj.strftime("%d/%m/%Y")
        print(formatted_date)
        sender_name = data.get("senderName")
        print(sender_name,"........................")
        selected_language = data.get("selectedLanguage")
        timestamp_str = data.get("timestamp")
        # timestamp = datetime.strptime(timestamp_str, "%d/%m/%Y, %H:%M:%S")
        try:
            timestamp = datetime.strptime(timestamp_str, "%d/%m/%Y, %H:%M:%S")
        except ValueError:
            # If the first format fails, try the second one
            timestamp = datetime.strptime(timestamp_str, "%m/%d/%Y, %I:%M:%S %p")
        print(timestamp,"....................timestamp......................")
        # Message object
        details = users_collection.find({"username":instructor_name})
        typing = data.get("action")
        print(typing,".../././/.//........")
        for i in details:
            if i["role"] == "instructor":
                if typing == "sent":
                    message_key_value = data.get("original_message")
                elif typing == "received":
                    message_key_value = data.get("translated_message")
                else:
                    message_key_value = None
                message = {
                    "sender_username": sender_name,
                    "message": message_key_value,
                    "translated_message": data.get("translated_message"),
                    "original_message": data.get("original_message"),
                    "action": data.get("action"),
                    "type": data.get("type"),
                    "timestamp": timestamp
                }
                # Check if document exists for room_id and instructor_name
                existing_document = chat_history_collection.find_one({"room_id": room_id, "instructor_name": instructor_name})
                print(room_id,instructor_name,"........////////...........")
                if existing_document:
                    print("//////////////////")
                    chat_history_collection.update_one(
                        {"room_id": room_id, "instructor_name": instructor_name, "date": formatted_date},
                        {"$push": {"chat_history": message}}
                    )
                else:
                    print(",,,,,,,,,,,,,,,,,,,,,,,,,,,,")
                    new_document = {
                        "room_id": room_id,
                        "instructor_name": instructor_name,
                        "selectedLanguage": selected_language,
                        "chat_history": [message],  # Initialize with the first message
                        "date": formatted_date,
                    }
                    chat_history_collection.insert_one(new_document)

                print("Chat history updated successfully.")
            else :
                print("He is not a Instructor",instructor_name)

    # pip install flask-socketio weasyprint pymongo

    @socketio.on("download_chat_history")
    def download_chat_history(data):
        print("Received data from client:", data)

        date_str = data.get("date")
        room_id = data.get("roomId")
        instructor_name = data.get("username")

        chatting = chat_history_collection.find({
            "room_id": room_id,
            "instructor_name": instructor_name
        })

        table_data = []
        sl_no = 1
        for chat in chatting:
            print(chat, "............DEBUG DATA.............")
            if 'chat_history' in chat:  
                for chat_item in chat['chat_history']:
                    action = chat_item.get("action")
                    o_message = chat_item.get("original_message", "")
                    t_message = chat_item.get("translated_message", "")
                    print(action,o_message,t_message,".....action, omessage, tmessage...................")
                    if action == "sent":
                        message = o_message
                    elif action == "received":
                        message = t_message
                    else:
                        message = ""
                    table_data.append({
                        "sl_no": sl_no,
                        "action": chat_item.get("action", ""),
                        "username": chat_item.get("sender_username", ""),
                        "message": message,
                        "type": chat_item.get("type", ""),
                    })
                    sl_no += 1

        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Chat History - {date_str}</title>
             <style>
                body {{
                    font-family: 'Arial', sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f4f4f9;
                    color: #333;
                }}
                .container {{
                    width: 90%;
                    margin: 20px auto;
                    padding: 30px;
                    background-color: white;
                    border-radius: 8px;
                    box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                    padding: 10px;
                    background-color: #4CAF50;
                    color: white;
                    border-radius: 8px;
                }}
                .header h1 {{
                    font-size: 36px;
                    margin: 0;
                }}
                .header p {{
                    font-size: 18px;
                    margin: 5px 0;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 20px;
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #4CAF50;
                    color: white;
                    font-size: 16px;
                }}
                td {{
                    font-size: 14px;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                tr:hover {{
                    background-color: #f1f1f1;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 30px;
                    font-size: 14px;
                    color: #777;
                }}
            </style>
        </head>
        <body>

        <div class="container">
            <div class="header">
                <h1>CHAT HISTORY</h1>
                <p>Date: {date_str}</p>
            </div>

            <table>
                <thead>
                    <tr>
                        <th>Sl.No</th>
                        <th>Action</th>
                        <th>Username</th>
                        <th>Message</th>
                        <th>Type</th>
                    </tr>
                </thead>
                <tbody>
        """

        # Add rows to the table
        for row in table_data:
            html_content += f"""
            <tr>
                <td>{row["sl_no"]}</td>
                <td>{row["action"]}</td>
                <td>{row["username"]}</td>
                <td>{row["message"]}</td>
                <td>{row["type"]}</td>
            </tr>
            """

        # Close the table and HTML tags
        html_content += """
                </tbody>
            </table>

            <div class="footer">
                <p>Copyright @2024- Powered by DIT / INICAI and Pinaca Technologies</p>
            </div>
        </div>

        </body>
        </html>
        """

        # Convert HTML to PDF
        pdf_data = weasyprint.HTML(string=html_content).write_pdf()

        # Send the PDF as a file stream to the client
        pdf_filename = f"chat_history_{date_str}.pdf"

        # Create a BytesIO stream for the PDF data
        pdf_stream = BytesIO(pdf_data)
        pdf_stream.seek(0)

        # Send the PDF to the client
        emit("pdf_download_ready", {"file_name": pdf_filename,"file_data": pdf_stream.read()})

        print(f"PDF ready for download: {pdf_filename}")

        # for captions download


    # @socketio.on('download_captions')
    # def handle_download_captions(data):
    #     print("Received data for captions download:", data)
    #     room_id = data.get("roomId")
    #     instructor_name = data.get("username")
    #     date = data.get("date")
    #     print(date,"....................")
    #     print("room_id", room_id)

    #     # Fetch data from MongoDB
    #     trans = room_sessions_collection.find({
    #         "room": room_id
    #     })
    #     trans_list = list(trans)
    #     # Prepare table data
    #     table_data = []
    #     sl_no = 1
    #     for caption in trans_list:
    #         instructor_lang_code = caption.get("instructor_lang")
    #         if 'transcriptions' in caption:  # Check if 'transcriptions' key exists
    #             for caption_item in caption['transcriptions']:
    #                 # Add transcription to the table
    #                 if instructor_lang_code == caption_item.get("src_language_code") and instructor_lang_code == caption_item.get("target_language_code"):
    #                     table_data.append({
    #                         "sl_no": sl_no,
    #                         "name": instructor_name,  # Use instructor_name for 'name'
    #                         "caption": caption_item.get("translated_text", "N/A")  # Default to "N/A" if key is missing
    #                     })
    #                     sl_no += 1

    #     # Generate styled HTML content
    #     html_content = f"""
    #     <!DOCTYPE html>
    #     <html lang="en">
    #     <head>
    #         <meta charset="UTF-8">
    #         <meta name="viewport" content="width=device-width, initial-scale=1.0">
    #         <title>Captions - {date}</title>
    #         <style>
    #             body {{
    #                 font-family: 'Arial', sans-serif;
    #                 background-color: #f4f4f9;
    #                 color: #333;
    #                 margin: 0;
    #                 padding: 20px;
    #             }}
    #             .container {{
    #                 max-width: 800px;
    #                 margin: auto;
    #                 background: white;
    #                 padding: 20px;
    #                 border-radius: 8px;
    #                 box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
    #             }}
    #             .header {{
    #                 text-align: center;
    #                 padding: 10px;
    #                 background-color: #4CAF50;
    #                 color: white;
    #                 border-radius: 8px;
    #             }}
    #             .header h1 {{
    #                 margin: 0;
    #                 font-size: 24px;
    #             }}
    #             table {{
    #                 width: 100%;
    #                 border-collapse: collapse;
    #                 margin-top: 20px;
    #             }}
    #             th, td {{
    #                 padding: 10px;
    #                 border: 1px solid #ddd;
    #                 text-align: left;
    #             }}
    #             th {{
    #                 background-color: #4CAF50;
    #                 color: white;
    #             }}
    #             tr:nth-child(even) {{
    #                 background-color: #f9f9f9;
    #             }}
    #         </style>
    #     </head>
    #     <body>
    #         <div class="container">
    #             <div class="header">
    #                 <h1>Captions</h1>
    #                 <p>Date: {date}</p>
    #             </div>
    #             <table>
    #                 <thead>
    #                     <tr>
    #                         <th>Sl. No</th>
    #                         <th>Name</th>
    #                         <th>Caption</th>
    #                     </tr>
    #                 </thead>
    #                 <tbody>
    #     """

    #     # Add rows dynamically to the HTML table
    #     for row in table_data:
    #         html_content += f"""
    #         <tr>
    #             <td>{row["sl_no"]}</td>
    #             <td>{row["name"]}</td>
    #             <td>{row["caption"]}</td>
    #         </tr>
    #         """

    #     # Close the HTML structure
    #     html_content += """
    #                 </tbody>
    #             </table>
    #             <div class="footer">
    #                 <p>Copyright @2024- Powered by DIT / INICAI and Pinaca Technologies</p>
    #             </div>
    #         </div>
    #     </body>
    #     </html>
    #     """

    #     # Convert HTML to PDF
    #     pdf_data = weasyprint.HTML(string=html_content).write_pdf()

    #     # Prepare the PDF file for download
    #     pdf_filename = f"captions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    #     # Send the PDF as a binary stream
    #     emit("pdf_download_ready", {
    #         "file_name": pdf_filename,
    #         "file_data": pdf_data
    #     })
    #     print(f"PDF ready for download: {pdf_filename}")


    @socketio.on('download_captions')
    def handle_download_captions(data):
        print("Received data for captions download:", data)
        room_id = data.get("roomId")
        instructor_name = data.get("username")
        date = data.get("date")
        print(date,"....................")
        print("room_id", room_id)

        # Fetch data from MongoDB
        trans = room_sessions_collection.find({
            "room": room_id
        })
        trans_list = list(trans)
        # Prepare table data
        table_data = []
        sl_no = 1
        for caption in trans_list:
            instructor_lang_code = caption.get("instructor_lang")
            if 'transcriptions' in caption:  # Check if 'transcriptions' key exists
                for caption_item in caption['transcriptions']:
                    table_data.append({
                        "sl_no": sl_no,
                        "role": caption_item.get("speaker_role", "N/A"),
                        "username": caption_item.get("speaker_username", "N/A"),
                        "caption" : caption_item.get("translated_text", "N/A"),
                    })
                    sl_no += 1

        # Generate styled HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Captions - {date}</title>
            <style>
                body {{
                    font-family: 'Arial', sans-serif;
                    background-color: #f4f4f9;
                    color: #333;
                    margin: 0;
                    padding: 20px;
                }}
                .container {{
                    max-width: 800px;
                    margin: auto;
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
                }}
                .header {{
                    text-align: center;
                    padding: 10px;
                    background-color: #4CAF50;
                    color: white;
                    border-radius: 8px;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 24px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 20px;
                }}
                th, td {{
                    padding: 10px;
                    border: 1px solid #ddd;
                    text-align: left;
                }}
                th {{
                    background-color: #4CAF50;
                    color: white;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Captions</h1>
                    <p>Date: {date}</p>
                    <p> Instructor Language Code - {instructor_lang_code}  </p>
                </div>
                <table>
                    <thead>
                        <tr>
                            <th>Sl. No</th>
                            <th>Role</th>
                            <th>Username</th>
                            <th>Caption</th>
                        </tr>
                    </thead>
                    <tbody>
        """

        # Add rows dynamically to the HTML table
        for row in table_data:
            html_content += f"""
            <tr>
                <td>{row["sl_no"]}</td>
                <td>{row["role"]}</td>
                <td>{row["username"]}</td>
                <td>{row["caption"]}</td>
            </tr>
            """

        # Close the HTML structure
        html_content += """
                    </tbody>
                </table>
                <div class="footer">
                    <p>Copyright @2024- Powered by DIT / INICAI and Pinaca Technologies</p>
                </div>
            </div>
        </body>
        </html>
        """

        # Convert HTML to PDF
        pdf_data = weasyprint.HTML(string=html_content).write_pdf()

        # Prepare the PDF file for download
        pdf_filename = f"captions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        # Send the PDF as a binary stream
        emit("pdf_download_ready", {
            "file_name": pdf_filename,
            "file_data": pdf_data
        })
        print(f"PDF ready for download: {pdf_filename}")



    @socketio.on('remove_participant')
    def handle_kick_out(data):
        user_id = data.get('user_id')
        room_id = data.get('room_id')
        role = data.get('role')
        
        print(user_id, room_id, "...................values_printing...................")
        print(rooms, "...............printing_rooms.........")

        if room_id in rooms:  # Check if the room exists
            if user_id in rooms[room_id]:  # Check if the user is in the room
                rooms[room_id].remove(user_id)  # Remove the user from the room
                print(f"User {user_id} removed from room {room_id}.")
                print(rooms, ".............after removed...............")
                
                # Notify remaining users in the room
                room_user_details = [user_session_details[uid] for uid in rooms[room_id]]
                socketio.emit(
                    "updateUsers",
                    {"users": rooms[room_id], "userDetails": room_user_details},
                    room=room_id,
                )

                # Notify the user that they were removed
                user_sid = user_sockets.get(user_id)
                if user_sid:
                    socketio.emit("kickedOut", {"message": "You have been removed by the instructor."}, to=user_sid)
                
                # Let the user leave the room
                leave_room(room_id, sid=user_sid)
                
                # Remove the room if empty
                if not rooms[room_id]:
                    del rooms[room_id]
                    print(f"Room {room_id} is now empty and has been removed.")
            else:
                print(f"User {user_id} not found in room {room_id}.")
        else:
            print(f"Room {room_id} does not exist.")


        
                