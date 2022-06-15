from django.urls import path

from .views import SignUpView, ProfileView, activate


urlpatterns = [
    path("signup/", SignUpView, name="signup"),
    path("profile/", ProfileView, name="profile"),
    path('activate/(?P<uidb64>[0-9A-Za-z_\-]+)/(?P<token>[0-9A-Za-z]{1,13}-[0-9A-Za-z]{1,20})/',  
        activate, name='activate'),  

    
]