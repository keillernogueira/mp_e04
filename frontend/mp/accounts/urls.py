from django.urls import path

from .views import ConfirmEmail, SignUpView, ProfileView, activate, ConfirmEmail, ValidatedEmail


urlpatterns = [
    path("signup/", SignUpView, name="signup"),
    path("profile/", ProfileView, name="profile"),
    path('activate/(<uidb64>/<token>/', activate, name='activate'),  
    path('confirm_email/', ConfirmEmail, name = 'ConfirmEmail'),
    path('validated_email/', ValidatedEmail, name = 'ValidatedEmail'),

    
]