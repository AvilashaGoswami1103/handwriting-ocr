#### TrOCR has been failing to detect messy doctor prescription handwriting. Switching to PaddleOCR.
#### While initializing PaddleOCR:
#####On the first run, PaddleOCR tries to automatically download several large model files (text detection, text recognition, etc.). This process can:
#####Take a long time (2–10 minutes)
#####Get stuck
#####Fail due to network issues, firewall, or slow internet
#####Show no clear error (just hangs)
### Switching to EasyOCR
