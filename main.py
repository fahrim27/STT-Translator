import os
from pydub import AudioSegment
import whisper
from faster_whisper import WhisperModel
from transformers import pipeline, AutoProcessor
# from pyannote.audio import Pipeline
# from utils import words_per_segment
# from pyannote_whisper.utils import diarize_text

from flask import *  
app = Flask(__name__)  


all_languages = ('afrikaans', 'af', 'albanian', 'sq', 'amharic', 'am', 'arabic', 'ar', 'armenian', 'hy', 'azerbaijani', 'az', 'basque', 'eu', 'belarusian', 'be','bengali', 'bn', 'bosnian', 'bs', 'bulgarian','bg', 'catalan', 'ca', 'cebuano','ceb', 'chichewa', 'ny', 'chinese (simplified)', 'zh-cn', 'chinese (traditional)','zh-tw', 'corsican', 'co', 'croatian', 'hr','czech', 'cs', 'danish', 'da', 'dutch','nl', 'english', 'en', 'esperanto', 'eo','estonian', 'et', 'filipino', 'tl', 'finnish','fi', 'french', 'fr', 'frisian', 'fy', 'galician', 'gl', 'georgian', 'ka', 'german','de', 'greek', 'el', 'gujarati', 'gu','haitian creole', 'ht', 'hausa', 'ha','hawaiian', 'haw', 'hebrew', 'he', 'hindi','hi', 'hmong', 'hmn', 'hungarian','hu', 'icelandic', 'is', 'igbo', 'ig', 'indonesian','id', 'irish', 'ga', 'italian','it', 'japanese', 'ja', 'javanese', 'jw','kannada', 'kn', 'kazakh', 'kk', 'khmer','km', 'korean', 'ko', 'kurdish (kurmanji)','ku', 'kyrgyz', 'ky', 'lao', 'lo','latin', 'la', 'latvian', 'lv', 'lithuanian','lt', 'luxembourgish', 'lb','macedonian', 'mk', 'malagasy', 'mg', 'malay','ms', 'malayalam', 'ml', 'maltese','mt', 'maori', 'mi', 'marathi', 'mr', 'mongolian','mn', 'myanmar (burmese)', 'my','nepali', 'ne', 'norwegian', 'no', 'odia', 'or','pashto', 'ps', 'persian', 'fa','polish', 'pl', 'portuguese', 'pt', 'punjabi','pa', 'romanian', 'ro', 'russian','ru', 'samoan', 'sm', 'scots gaelic', 'gd','serbian', 'sr', 'sesotho', 'st','shona', 'sn', 'sindhi', 'sd', 'sinhala', 'si','slovak', 'sk', 'slovenian', 'sl','somali', 'so', 'spanish', 'es', 'sundanese','su', 'swahili', 'sw', 'swedish','sv', 'tajik', 'tg', 'tamil', 'ta', 'telugu','te', 'thai', 'th', 'turkish','tr', 'ukrainian', 'uk', 'urdu', 'ur', 'uyghur','ug', 'uzbek', 'uz','vietnamese', 'vi', 'welsh', 'cy', 'xhosa', 'xh','yiddish', 'yi', 'yoruba','yo', 'zulu', 'zu')

@app.route('/')  
def upload():  
    return render_template("file_upload_form.html")  

@app.route('/translate', methods = ['POST'])  
def success():  
    if request.method == 'POST':
        PATH = 'static/files/input/'
        PATH2 = 'static/files/output/'

        f = request.files['file']
        FILE_NAME = f.filename 
        f.save(os.path.join(PATH,FILE_NAME))
        
        if os.path.splitext(f.filename)[1] != '.wav':
            prepare_voice_file(os.path.join(PATH,f.filename))

        input_path = os.path.join(PATH,os.path.splitext(f.filename)[0]+'.wav')
        output_path = os.path.join(PATH2,os.path.splitext(f.filename)[0]+'.txt')
        
        results = speech_to_text(input_path, output_path, request.form.get('type'))
        
        return render_template("success.html", results=results)

def prepare_voice_file(path):
    """
    Converts the input audio file to WAV format if necessary and returns the path to the WAV file.
    """
    if os.path.splitext(path)[1] in ('.mp3', '.m4a', '.ogg', '.flac'):
        audio_file = AudioSegment.from_file(path, format=os.path.splitext(path)[1][1:])
        wav_file = os.path.splitext(path)[0] + '.wav'
        audio_file.export(wav_file, format='wav')
        return wav_file
    else:
        raise ValueError(
            f'Unsupported audio format: {format(os.path.splitext(path)[1])}')


def transcribe_audio(audio_data):
    """
    Transcribes audio data to text using Google's speech recognition API.
    """
    # pipeline = Pipeline.from_pretrained(
    #     "pyannote/speaker-diarization-3.1", use_auth_token="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"
    # )

    # model = whisper.load_model("medium")
    # diarization_result = pipeline(audio_data)
    # transcription_result = model.transcribe(audio_data, word_timestamps=True)

    # final_result = words_per_segment(transcription_result, diarization_result)

    # for _, segment in final_result.items():
    #     print(f'{segment["start"]:.3f}\t{segment["end"]:.3f}\t {segment["speaker"]}\t{segment["text"]}')

    # transcriber = pipeline(
    #   "automatic-speech-recognition", 
    #   model="cahya/whisper-small-id"
    # )
    # transcriber.model.config.forced_decoder_ids = (
    #   transcriber.tokenizer.get_decoder_prompt_ids(
    #     language="id", 
    #     task="transcribe"
    #   )
    # )
    # transcription_result = transcriber(audio_data)

    # model_id = "distil-whisper/distil-large-v2"
    # model_path = Path(model_id)
    # if not model_path.exists():
    #     ov_model = OVModelForSpeechSeq2Seq.from_pretrained(
    #         model_id, export=True, compile=False, load_in_8bit=False)
    #     ov_model.half()
    #     ov_model.save_pretrained(model_path)
    # else:
    #     ov_model = OVModelForSpeechSeq2Seq.from_pretrained(
    #         model_path, compile=False)

    # processor = AutoProcessor.from_pretrained(model_id)

    # ov_model.to("AUTO")
    # ov_model.compile()

    # # ... load input audio and reference text
    # input_features = processor(input_audio).input_features
    # predicted_ids = ov_model.generate(input_features)
    # transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    # print(f"Reference: {reference_text}")
    # print(f"Result: {transcription}")

    # pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
    #                                     use_auth_token="your/token")
    # model = whisper.load_model("tiny")
    # asr_result = model.transcribe(audio_data)
    # diarization_result = pipeline(audio_data)
    # final_result = diarize_text(asr_result, diarization_result)

    # for seg, spk, sent in final_result:
    #     line = f'{seg.start:.2f} {seg.end:.2f} {spk} {sent}'
    #     print(line)

    model = WhisperModel("cahya/faster-whisper-medium-id")

    segments, info = model.transcribe(audio_data, language="id", condition_on_previous_text=False)
    result = ""
    for segment in segments:
        result + "[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text)

    return result

def transcribe_audio_fulltext(audio_data):
    """
    Transcribes audio data to text using Google's speech recognition API.
    """
    mymodel = whisper.load_model("small")
    result = mymodel.transcribe(audio_data)

    # audio = whisper.load_audio(audio_data)
    # audio = whisper.pad_or_trim(audio)

    # mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # _, probs = model.detect_language(mel)
    # print(f"Detected language: {max(probs, key=probs.get)}")

    # options = whisper.DecodingOptions()
    # result = whisper.decode(model, mel, options)
    
    # print(result["text"])
    # model_size = "large-v3"

    # model = WhisperModel(model_size, device="cuda", compute_type="float16")

    # segments, info = model.transcribe(
    #     "audio.mp3",
    #     beam_size=5,
    #     vad_filter=True,
    #     vad_parameters=dict(min_silence_duration_ms=500),
    # )

    # print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    # text = ""
    # for segment in segments:
    #     text + "[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text)
    return result["text"]

def write_transcription_to_file(text, output_file):
    """
    Writes the transcribed text to the output file.
    """
    with open(output_file, 'w') as f:
        f.write(text)


def speech_to_text(input_path: str, output_path: str, type):
    """
    Transcribes an audio file at the given path to text and writes the transcribed text to the output file.
    """
    if type == 1:
        text = transcribe_audio(input_path)
    else:
        text = transcribe_audio_fulltext(input_path)
    
    print(text)
    return text

if __name__ == '__main__':  
    app.run(host= '127.0.0.1', debug = True)  
