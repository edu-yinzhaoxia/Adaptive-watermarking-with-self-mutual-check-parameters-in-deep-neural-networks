1. Prepare your own model (trained)

2. Load into 2uexchange.py and embed the check bits between parameters (multiple check bits)

3. Load the obtained model with embedded watermark into simple.adaptivechange-2.py for adaptive improvement

4. Embed the self check bit of the adaptively enhanced watermark through mod3. py

5. If you need to attack the model, you can use random-attack.py to adjust the parameters

6. CheckeAndrecovery.py can be used to recover the attacked model




Note:

Utils.py is the function tool library used

Ceshi_lenet.py is the final function for testing accuracy, simply fill the model path into the root.