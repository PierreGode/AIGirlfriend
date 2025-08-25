# AIGirlfriend

Voice AI girlfriend powered by `gpt-5-nano` with streaming text and speech.

Clone repo to Windows machine
```
git clone https://github.com/PierreGode/AIGirlfriend.git
```
```
cd AIGirlfriend
```
```
pip install -r requirements.txt
```
Set AI key in Enviroment
``` 
$env:OPENAI_API_KEY="sk-proj-"
```

Edit `aigirlfriend.py` and set your name by changing `"Love"`.

The assistant now uses the unified `gpt-5-nano` voice model to stream
text and audio in a single request for lower latency and cost.
