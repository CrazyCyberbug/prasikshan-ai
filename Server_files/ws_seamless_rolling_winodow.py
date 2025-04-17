from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import threading
import numpy as np
import json

app = FastAPI()
from fastapi import FastAPI, Request
from pydantic import BaseModel
import numpy as np
import uvicorn
import json
import time
import threading
import torch
from typing import List, Dict
from transformers import AutoProcessor, SeamlessM4Tv2Model, VitsModel, AutoTokenizer
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

vad_model = load_silero_vad()

app = FastAPI()

# Define constants
sampling_rate = 16000
batch_size = sampling_rate * 3
keep_samples = int(sampling_rate * 0.15)
max_q_size = 50 * batch_size
is_running = True

# Initialize model and processor
processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
model.to('cuda')
model.eval()

# TTS model for kaz and mya
tts_speech_model = None
tts_lang = None
tts_tokenizer = None


# Global dictionaries for speech translation
speech_queues: Dict[str, np.ndarray] = {}
total_speech_queues: Dict[str, np.ndarray] = {}
speech_response_lists: Dict[str, List] = {}
speech_buffer_locks: Dict[str, threading.Lock] = {}
speech_response_list_locks: Dict[str, threading.Lock] = {}
speech_client_langs: Dict[str, str] = {}
speech_inference_threads: Dict[str, threading.Thread] = {}
speech_client_active: Dict[str, bool] = {}
speech_thread_locks: Dict[str, threading.Lock] = {}


prev_len = 0
full_transcript = ""
prev_transcript = ""

class AudioData(BaseModel):
    """
    Summary: Template for recieving audio data from backend.
    """
    client_id: str
    audio_data: List[float]
    sampling_rate: int
    tgt_lang: str
    reset_buffers: str

def calculate_rms_energy(audio_chunk):
    return np.sqrt(np.mean(np.square(audio_chunk)))
#  Using rms energy for seperating white noise from audio. 
# def is_silence(audio_chunk, silence_threshold=0.01):
#     rms_energy = calculate_rms_energy(audio_chunk)
#     print(f"RMS Energy: {rms_energy}")
#     return rms_energy < silence_threshold

def do_vad(wav):
    """
    Summary:  Use silero VAD to detect if audio chunk contains speech.
    Non speech segments are directly discarded. 
    """
    speech_timestamps = get_speech_timestamps(wav,
                                            vad_model,
                                            return_seconds=True,
                                            threshold = 0.50)  # Return speech timestamps in seconds (default is samples)
    return speech_timestamps

def is_silence(audio_chunk, silence_threshold=0.01):
    """
    Summary: VAD module to check if speech is there in the chunk.
    """
    print("detected silence")
    return len(do_vad(audio_chunk)) == 0

def speech2speech(audio_inputs, tgt_lang):
    """
    Summary: Method to translate speech to speech for languages,
    that do not need any intermediate processing - English, French, Vietnmese.
    """
    speech = model.generate(**audio_inputs, tgt_lang=tgt_lang)[0].cpu().numpy().squeeze()
    return speech

def load_TTS_model(tgt_lang_code):
    """
    Load language specific vits - TTS model.
    """
    global tts_speech_model, tts_lang, tts_tokenizer
    
    if tts_speech_model == None or tts_lang != tgt_lang_code:        
        tts_lang = tgt_lang_code
        tts_speech_model = VitsModel.from_pretrained(f"facebook/mms-tts-{tgt_lang_code}").to('cuda')
        tts_tokenizer = AutoTokenizer.from_pretrained(f"facebook/mms-tts-{tgt_lang_code}")

    return tts_speech_model, tts_tokenizer    

def transcribe_speech(input_array, tgt_lang):
    audio_inputs = processor(audios=input_array, sampling_rate=16000, return_tensors="pt").to('cuda')
    output_tokens = model.generate(**audio_inputs, tgt_lang=tgt_lang, generate_speech=False)
    translated_text_from_audio = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)

    if tgt_lang in ['mya', 'kaz']:
        speech_model, tokenizer = load_TTS_model(tgt_lang)
        inputs = tokenizer(translated_text_from_audio, return_tensors="pt").to('cuda')

        with torch.no_grad():
            output = speech_model(**inputs).waveform
        speech = output.squeeze().cpu().numpy()
    else:
        speech = speech2speech(audio_inputs=audio_inputs, tgt_lang=tgt_lang)
    
    return translated_text_from_audio, speech.tolist()

def transcribe_speech(input_array, tgt_lang):
    print("inside transcribe speech")
    try:
        # get input_tokens
        audio_inputs = processor(audios=input_array, sampling_rate=16000, return_tensors="pt").to('cuda')

        if tgt_lang in ['mya', 'kaz']:
            # generate text in lang.
            output_tokens = model.generate(**audio_inputs, tgt_lang=tgt_lang, generate_speech=False)
            translated_text_from_audio = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
            
            # Clear GPU memory
            del audio_inputs
            del output_tokens
            torch.cuda.empty_cache()
            
            #generate speech.
            speech_model, tokenizer = load_TTS_model(tgt_lang)
            inputs = tokenizer(translated_text_from_audio, return_tensors="pt").to('cuda')
            with torch.no_grad():
                output = speech_model(**inputs).waveform
            speech = output.squeeze().cpu().numpy()
            
            # Clear GPU memory
            del inputs
            del output
            torch.cuda.empty_cache()
            
        elif tgt_lang in ['hin', 'ben', 'arb']:
            # generate intermediate eng text.
            output_tokens = model.generate(**audio_inputs, tgt_lang=tgt_lang, generate_speech=False)
            eng_text = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
            
            # Clear GPU memory
            del audio_inputs
            del output_tokens
            torch.cuda.empty_cache()
            
            # generate text and speech in tgt_lang.
            audio_inputs = processor(text=eng_text, sampling_rate=16000, return_tensors="pt").to('cuda')
            output_tokens = model.generate(**audio_inputs, tgt_lang=tgt_lang, return_intermediate_token_ids=True)
            speech = output_tokens.waveform.cpu().numpy()
            translated_text_from_audio = processor.decode(output_tokens.sequences.tolist()[0], skip_special_tokens=True)
            
            # Clear GPU memory
            del audio_inputs
            del output_tokens
            torch.cuda.empty_cache()
            
        else:
            # generate text and speech in tgt_lang.
            output_tokens = model.generate(**audio_inputs, tgt_lang=tgt_lang, return_intermediate_token_ids=True)
            speech = output_tokens.waveform.cpu().numpy()
            translated_text_from_audio = processor.decode(output_tokens.sequences.tolist()[0], skip_special_tokens=True)
            
            # Clear GPU memory
            del audio_inputs
            del output_tokens
            torch.cuda.empty_cache()
        
        return translated_text_from_audio, speech.flatten().tolist()
    
    except Exception as e:
        print(f"Error in transcribe_speech: {e}")
        torch.cuda.empty_cache()  # Clean up even in case of error
        raise

def batched_transcribe_speech(batched_input_array, tgt_lang):
    print("inside batched trancribe speech")
    try:
        audio_inputs = processor(audios=batched_input_array, sampling_rate=16000, return_tensors="pt").to('cuda')
        
        if tgt_lang in ['mya', 'kaz']:
            # generate text in lang.
            output_tokens = model.generate(**audio_inputs, tgt_lang=tgt_lang, generate_speech=False)
            translated_text_from_audio = " ".join([processor.decode(sequence.tolist(),
                                                                skip_special_tokens=True)
                                               for sequence in output_tokens.sequences])
            
            # Clear GPU memory
            del audio_inputs
            del output_tokens
            torch.cuda.empty_cache()
            
            #generate speech.
            speech_model, tokenizer = load_TTS_model(tgt_lang)
            inputs = tokenizer(translated_text_from_audio, return_tensors="pt").to('cuda')
            with torch.no_grad():
                output = speech_model(**inputs).waveform
            speech = output.squeeze().cpu().numpy()
            
            # Clear GPU memory
            del inputs
            del output
            torch.cuda.empty_cache()
        
        elif tgt_lang in ["hin", "ben"]:
            output_tokens = model.generate(**audio_inputs, tgt_lang="eng", generate_speech=False)
            translated_text_from_audio = " ".join([processor.decode(sequence.tolist(),
                                                                skip_special_tokens=True)
                                               for sequence in output_tokens.sequences])
            
            # Clear GPU memory
            del audio_inputs
            del output_tokens
            torch.cuda.empty_cache()
            
            audio_inputs = processor(text=translated_text_from_audio, sampling_rate=16000, return_tensors="pt").to('cuda')
            output_tokens = model.generate(**audio_inputs, tgt_lang=tgt_lang, return_intermediate_token_ids=True)
            speech = output_tokens.waveform.cpu().numpy()
            translated_text_from_audio = " ".join([processor.decode(sequence.tolist(),
                                                                    skip_special_tokens=True)
                                                for sequence in output_tokens.sequences])
            
            # Clear GPU memory
            del audio_inputs
            del output_tokens
            torch.cuda.empty_cache()
        
        else:
            output_tokens = model.generate(**audio_inputs, tgt_lang=tgt_lang, return_intermediate_token_ids=True)
            speech = output_tokens.waveform.cpu().numpy()
            translated_text_from_audio = " ".join([processor.decode(sequence.tolist(),
                                                                    skip_special_tokens=True)
                                                for sequence in output_tokens.sequences])
            
            # Clear GPU memory
            del audio_inputs
            del output_tokens
            torch.cuda.empty_cache()
            
        return translated_text_from_audio, speech.flatten().tolist()
    
    except Exception as e:
        print(f"Error in batched_transcribe_speech: {e}")
        torch.cuda.empty_cache()  # Clean up even in case of error
        raise
# old code for using 
def speech_inference(client_id):
    """
    Summary: This is the main method defining the logic flow.
    This function keeps polling  in_buffers for availability of audio chunks in the audio buffers, 
    If available, the chunks are processed and results are pushed into out_buffers.

    if the available audio is of length

    3 - 5 seconds - the audio is directly processed.
    5 - 15 seconds - audio is batched with  batch size of 3 seconds i.e   12 second --> [(3 sec), (3 sec), (3 sec)]

    """
    while speech_client_active.get(client_id, False):
        # print(f"[{client_id}] has {len(speech_queues[client_id]) / 16000 :.1f} seconds of unprocessed audio")
        with speech_thread_locks[client_id]:
            tgt_lang = speech_client_langs.get(client_id)
            if not tgt_lang:
                continue
               
            queue = speech_queues.get(client_id)
            
            if queue is None or queue.size < 3 *sampling_rate:
                continue
                        
      
            MIN_INPUT_SIZE = 3 * sampling_rate  # proceesing a minimum of 3 seconds of audio
            MAX_NON_BATCHED_INPUT_SIZE = 5 * sampling_rate
            MAX_INPUT_SIZE = 15 * sampling_rate # processing a maximum of 12 seconds of audio
            
            try:
                print(f"\n\nThe total audio  now contains {len(total_speech_queues.get(client_id))/16_000: .2f} seconds of audio.\n\n")
                if queue.size >= MIN_INPUT_SIZE and queue.size <= MAX_NON_BATCHED_INPUT_SIZE:
                    start = time.time()
                    audio_chunk = queue[:]
                    response = transcribe_speech(audio_chunk, tgt_lang)
                    speech_queues[client_id] = queue[ -keep_samples:]
                    end = time.time()
                    print(f"[Simple] Took {end  - start} seconds to process audio of  {len(audio_chunk) / sampling_rate}")
                elif queue.size > MAX_NON_BATCHED_INPUT_SIZE:
                    batch_size = min(MAX_INPUT_SIZE, len(queue))
                    print(f"processing an input of {batch_size / sampling_rate:.1f} seconds")
                    
                    batched_audio_chunk = [queue[i: i + MIN_INPUT_SIZE] for i in range(0, batch_size, MIN_INPUT_SIZE)]
                    response = batched_transcribe_speech(batched_input_array = batched_audio_chunk, tgt_lang = tgt_lang)
                    speech_queues[client_id] = queue[batch_size - keep_samples:]
                
                torch.cuda.empty_cache()
                with speech_response_list_locks[client_id]:
                    print(response[0])
                    speech_response_lists[client_id].append(response)
                
                print(f"[{client_id}] has {len(speech_queues[client_id]) / 16000 :.1f} seconds of unprocessed audio")
                
            except Exception as e:
                print(f"Error processing audio for client {client_id}: {e}")

# code to do batch processing using Seamless-m4t-v2 using full audio. 
def transcribe_smls(total_audio, tgt_lang = "eng"):
    input_array = [total_audio[i : i + 5* 16_000] for i in range(0, len(total_audio), 5 * 16_000)]
    audio_inputs = processor(audios=input_array, src_lang= "hin",  sampling_rate=16000, return_tensors="pt").to('cuda')
    output_tokens = model.generate(**audio_inputs, tgt_lang=tgt_lang, return_intermediate_token_ids=True)
    
    
    # speech = output_tokens.waveform.cpu().numpy()
    translated_text_from_audio = " ".join([processor.decode(sequence.tolist(), skip_special_tokens=True)
                                        for sequence in output_tokens.sequences])
    
    audio_inputs = audio_inputs.to('cpu')
    del audio_inputs
    del output_tokens
    return translated_text_from_audio, []

# code to usetotal audio in transcribing.
def speech_inference(client_id):
    """
    Summary: This is the main method defining the logic flow.
    This function keeps polling  in_buffers for availability of audio chunks in the audio buffers, 
    If available, the chunks are processed and results are pushed into out_buffers.

    if the available audio is of length

    3 - 5 seconds - the audio is directly processed.
    5 - 15 seconds - audio is batched with  batch size of 3 seconds i.e   12 second --> [(3 sec), (3 sec), (3 sec)]

    """
    global full_transcript, prev_transcript, prev_len
    while speech_client_active.get(client_id, False):
        # print(f"[{client_id}] has {len(speech_queues[client_id]) / 16000 :.1f} seconds of unprocessed audio")
        with speech_thread_locks[client_id]:
            tgt_lang = speech_client_langs.get(client_id)
            if not tgt_lang:
                continue
               
            queue = speech_queues.get(client_id)
            
            if queue is None or queue.size < 3 *sampling_rate:
                continue
                        
      
            MIN_INPUT_SIZE = 3 * sampling_rate  # proceesing a minimum of 3 seconds of audio
            MAX_NON_BATCHED_INPUT_SIZE = 5 * sampling_rate
            MAX_INPUT_SIZE = 15 * sampling_rate # processing a maximum of 12 seconds of audio
            
            try:
                print(f"\n\nThe total audio  now contains {len(total_speech_queues.get(client_id))/16_000: .2f} seconds of audio.\n\n")
                
                # clearing the queue.
                audio_chunk = queue[:]
                speech_queues[client_id] = queue[ -keep_samples:]
                
                # Truncate if audio >=30 seconds 
                audio = total_speech_queues.get(client_id)[:].copy()
                if len(audio) >= 30 * 16_000:
                    total_speech_queues[client_id] = audio[prev_len:]
                    full_transcript = prev_transcript
                    print("30 second mark hit resetting window")
                    
                start = time.time()
                # audio = total_speech_queues[client_id][:].copy()
                response = transcribe_smls(total_audio = audio, tgt_lang =  tgt_lang)
                text, translated_speech = response
                end = time.time()
                
                print(f"inference took {end - start :.2f} seconds.")

                current_full_transcript =  full_transcript + " " + text
                response = current_full_transcript, translated_speech
                
                prev_len = len(audio)
                prev_transcript = current_full_transcript

                
                
                print(f"Transcriptions: {text}")
                
                torch.cuda.empty_cache()
                
                # storing transcription.
                with speech_response_list_locks[client_id]:
                    print(response[0])
                    speech_response_lists[client_id].append(response)
                
                print(f"[{client_id}] has {len(speech_queues[client_id]) / 16000 :.1f} seconds of unprocessed audio")
                
            except Exception as e:
                print(f"Error processing audio for client {client_id}: {e}")



# WebSocket endpoint for handling audio packets and translations
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    client_id = None
    tgt_lang = None

    try:
        while True:
            # Receive message (audio packet)
            data = await websocket.receive_text()
            audio_data = json.loads(data)
            # 
            client_id = audio_data.get("client_id")
            audio =  audio_data.get("audio_data")
            tgt_lang = audio_data.get("tgt_lang")
            sampling_rate  = audio_data.get("sampling_rate")
            tgt_lang  = audio_data.get("tgt_lang")
            reset_buffers = audio_data.get("reset_buffers")

            print(client_id, len(audio), tgt_lang, sampling_rate, tgt_lang, reset_buffers)


            if client_id not in speech_queues:
                # Initialize data structures for the client
                speech_queues[client_id] = np.ndarray([], np.float32)
                total_speech_queues[client_id] = np.ndarray([], np.float32)
                speech_response_lists[client_id] = []
                speech_buffer_locks[client_id] = threading.Lock()
                speech_response_list_locks[client_id] = threading.Lock()
                speech_thread_locks[client_id] = threading.Lock()
                speech_client_active[client_id] = True
                
                # Start the inference thread for the client
                speech_inference_threads[client_id] = threading.Thread(
                    target=speech_inference,
                    args=(client_id,),
                    daemon=True 
                )
                speech_inference_threads[client_id].start()

            # Handle target language change
            with speech_thread_locks[client_id]:
                speech_client_langs[client_id] = tgt_lang

            if audio_data.get("reset_buffers", "false").lower() == "true":
                # Reset the buffers
                with speech_thread_locks[client_id]:
                    print(f"Resetting buffers for client: {client_id}")
                    speech_response_lists[client_id] = []
                    speech_queues[client_id] = np.ndarray([], np.float32)

            # Add the received audio data to the queue
            with speech_buffer_locks[client_id]:
                audio_chunk = np.array(audio_data.get("audio_data", []))
                if speech_queues[client_id].size < max_q_size and not is_silence(audio_chunk):
                    speech_queues[client_id] = np.append(speech_queues[client_id], audio_chunk)
                    total_speech_queues[client_id] = np.append(total_speech_queues[client_id], audio_chunk)

            # If there are responses ready, send them back to the client
            transcriptions = ""
            with speech_response_list_locks[client_id]:
                if speech_response_lists[client_id]:
                    if len(speech_response_lists[client_id])> 0:                
                        transcriptions = speech_response_lists[client_id].pop(0)

            # Send the response back to the client (via WebSocket)
            await websocket.send_text(json.dumps({
                "status": "Audio received",
                "processed": len(audio_data["audio_data"]),
                "transcriptions": str(transcriptions),
                "sent_timstamp":  time.time()

            }))

    except WebSocketDisconnect:
        print(f"Client {client_id} disconnected")
        speech_client_active[client_id] = False

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004, ws_max_size= 10*1024*1024)
