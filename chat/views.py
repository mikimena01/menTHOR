from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, redirect

from menTHOR.utils import get_answer
from .models import Message

from django.shortcuts import render, redirect
from django.utils.crypto import get_random_string
from .models import Message

def chat_view(request):
    # Se l'utente non ha una sessione, creiamo un ID casuale
    if 'session_key' not in request.session:
        request.session['session_key'] = get_random_string(32)  # ID univoco per la sessione
        request.session.modified = True  # Assicura che venga salvato il cookie

    session_key = request.session['session_key']

    if request.method == 'POST':
        user_message = request.POST.get('message')
        Message.objects.create(text=user_message, is_bot=False, session_key=session_key)

        # Simula una risposta del bot
        text, document = get_answer(user_message)
        Message.objects.create(text=text, reference=document, is_bot=True, session_key=session_key)

        return redirect('chat')

    # Filtra i messaggi solo per la sessione attuale
    messages = Message.objects.filter(session_key=session_key).order_by('timestamp')
    return render(request, 'chat.html', {'messages': messages})
