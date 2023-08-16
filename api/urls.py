from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .viewsets import AgencyViewset, DataViewset, login_view, register
from django.contrib import admin
from django.views.decorators.csrf import csrf_exempt

router = DefaultRouter()
router.register(r'agencies', AgencyViewset, basename='agency')
router.register('data', DataViewset)


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include(router.urls)),
    path('login/', login_view),
    path('register/', csrf_exempt(register)),
]
