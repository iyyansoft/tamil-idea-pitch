from deep_translator import GoogleTranslator


class TamilTranslator:
    def __init__(self):
        self.translator = GoogleTranslator(source='en', target='ta')

    def translate(self, text):
        try:
            return self.translator.translate(text)
        except Exception as e:
            print("[ERROR]", e)
            return text