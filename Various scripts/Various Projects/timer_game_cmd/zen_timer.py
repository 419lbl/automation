from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
import time

TARGET, TOL = 10.0, 0.05
start = None
running = False

class ZenTimer(FloatLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.label = Label(text='Press', font_size='40sp',
                           pos_hint={'center_x':0.5, 'center_y':0.6})
        self.btn = Button(text='Start / Stop', size_hint=(0.4,0.2),
                          pos_hint={'center_x':0.5,'center_y':0.3})
        self.btn.bind(on_press=self.toggle)
        self.add_widget(self.label)
        self.add_widget(self.btn)

    def toggle(self, *_):
        global running, start
        if not running:
            start = time.perf_counter()
            running = True
            self.label.text = '...'
        else:
            d = (time.perf_counter() - start) - TARGET
            running = False
            self.label.text = ("🎯" if abs(d) <= TOL else "❌") + f" Δ{d:+.3f}s"

class ZenApp(App):
    def build(self):
        self.title = "Zen Timer"
        return ZenTimer()

if __name__ == '__main__':
    ZenApp().run()
