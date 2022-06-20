from attr import attr
from django import forms  
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm  
from django.core.exceptions import ValidationError  
from django.forms.fields import EmailField  
from django.forms.forms import Form  
  
class CustomUserCreationForm(UserCreationForm):

    username = forms.CharField(label='username', min_length=5, max_length=150)  
    email = forms.EmailField(label='email')  
    password1 = forms.CharField(label='password', min_length=8, widget=forms.PasswordInput)  
    password2 = forms.CharField(label='Confirm password', min_length=8, widget=forms.PasswordInput)  

  
    def email_clean(self):
        print("email_clean")
        emaill = self.cleaned_data['email'].lower()  
        new = User.objects.filter(email=emaill)
        print(User.objects)
        if new.count():  
            self.add_error('email', "Email já Cadastrado.")
            return False
        return emaill  


    def clean_password2(self): 
        print("password clean")
        password1 = self.cleaned_data['password1']  
        password2 = self.cleaned_data['password2']  
  
        if password1 and password2 and password1 != password2:  
            raise ValidationError("Password don't match")  
        return password2  
  
    def save(self, commit = True):  
        user = User.objects.create_user(  
            self.cleaned_data['username'],  
            self.cleaned_data['email'],  
            self.cleaned_data['password1']  
        )  
        return user  