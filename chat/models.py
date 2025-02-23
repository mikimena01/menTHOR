from django.db import models

class Message(models.Model):
    text = models.TextField()
    is_bot = models.BooleanField(default=False)
    timestamp = models.DateTimeField(auto_now_add=True)
    reference = models.TextField(default="")
    session_key = models.CharField(max_length=32, default='NONE')  # Per filtrare i messaggi per sessione


    def __str__(self):
        return self.text