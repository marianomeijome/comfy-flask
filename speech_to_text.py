from faster_whisper import WhisperModel
import os
import pyaudio
import wave

def record_chunk(p, stream, file_path, chunk_length=1):
    frames = []
    for _ in range(0, int(16000 / 1024 * chunk_length)):
        data = stream.read(1024)
        frames.append(data)
    
    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()

def transcribe_chunk(model, file_path):
    segments, _ = model.transcribe(file_path, beam_size=5)
    return ''.join(segment.text for segment in segments)

def transcribe_audio():
    # Choose your model settings
    model_size = "medium.en"
    model = WhisperModel(model_size, device="cuda", compute_type="float32")

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)

    accumulated_transcription = "" # Initialize an empty string to accumulate transcription

    try:
        while True:
            chunk_file = "temp_chunk.wav"
            record_chunk(p, stream, chunk_file)
            transcription = transcribe_chunk(model, chunk_file)
            print(transcription)
            os.remove(chunk_file)

            # Append the new transcription to the accumulated transcription
            accumulated_transcription += transcription

            # Print the accumulated transcription
            #print(accumulated_transcription)

    except KeyboardInterrupt:
        print("Transcription stopped by user")
        # Write the accumulated transcription to a file
        with open("log.txt", "w") as log_file:
            log_file.write(accumulated_transcription)

    finally:
        print("Stopping stream...")
        stream.stop_stream()
        stream.close()
        p.terminate()
    
