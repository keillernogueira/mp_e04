from django.contrib.auth import forms  
from django.shortcuts import redirect, render  
from django.contrib import messages  
from django.contrib.auth.forms import UserCreationForm  
from .forms import CustomUserCreationForm  
from django.http import HttpResponse, HttpResponseRedirect
from django.contrib.auth import authenticate, login

from django.http import HttpResponse  
from django.shortcuts import render, redirect  
from django.contrib.auth import login, authenticate  
from django.utils.encoding import force_bytes, force_str
from django.contrib.sites.shortcuts import get_current_site  
from django.template.loader import render_to_string 
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.core.mail import EmailMessage  
from .tokens import account_activation_token  
from django.contrib.auth.models import User  

from django.contrib.messages import get_messages
from django.contrib.auth import get_user_model


# Create your views here.  
  

def SignUpView(request): 

    if request.method == 'POST':  
        form = CustomUserCreationForm(request.POST)  
        if form.is_valid() and form.email_clean():
            '''print("valido")
            form.save()
            messages.success(request, 'Usuário Criado com Sucesso.')
            new_user = authenticate(username=form.cleaned_data['username'],
                                    password=form.cleaned_data['password1'],
                                    )
            login(request, new_user)
            return HttpResponseRedirect("/e04")'''

            user = form.save(commit=False)  
            user.is_active = False  
            user.save()  
            #This is  to obtain the current cite domain   
            current_site_info = get_current_site(request)  
            mail_subject = 'The Activation link has been sent to your email address'  
            message = render_to_string('registration/acc_active_email.html', {  
                'user': user,  
                'domain': current_site_info.domain,  
                'uid':urlsafe_base64_encode(force_bytes(user.pk)),  
                'token':account_activation_token.make_token(user),  
            })  
            to_email = form.cleaned_data.get('email')  
            email = EmailMessage(  
                        mail_subject, message, to=[to_email]  
            )
            email.send()
            request.session['username'] = user.username
            return HttpResponseRedirect("/accounts/confirm_email")

        else:
            print('invalido')
            print(form.errors)
            messages.error(request, "Falha ao criar usuário.") 
    else: 
        print("new get")
        form = CustomUserCreationForm()
    context = {  
        'form':form  
    }  
    return render(request, 'registration/signup.html', context)


def ConfirmEmail(request):
    print(request.session['username'])

    if request.method == 'POST':
        username = request.session['username']
        u = get_user_model()
        user = u.objects.get(username = username)
        current_site_info = get_current_site(request)  
        mail_subject = 'The Activation link has been sent to your email address'  
        message = render_to_string('registration/acc_active_email.html', {  
            'user': user,  
            'domain': current_site_info.domain,  
            'uid':urlsafe_base64_encode(force_bytes(user.pk)),  
            'token':account_activation_token.make_token(user),  
        })  
        to_email = user.email  
        email = EmailMessage(  
                    mail_subject, message, to=[to_email]  
        )
        email.send()
        return HttpResponseRedirect("/accounts/confirm_email")

    return render(request, 'registration/confirm_email.html')

def ValidatedEmail(request):
    return render(request, 'registration/validated_email.html')



def activate(request, uidb64, token):  

    user = get_user_model()  
    try:  
        uid = force_str(urlsafe_base64_decode(uidb64))  
        user = User.objects.get(pk=uid)  
    except(TypeError, ValueError, OverflowError, User.DoesNotExist):  
        user = None  
    if user is not None and account_activation_token.check_token(user, token):  
        user.is_active = True  
        user.save()  
        return HttpResponseRedirect("/accounts/validated_email")  
    else:  
        return HttpResponse('Activation link is invalid!')  

def ProfileView(request):
    return HttpResponseRedirect("/e04")
