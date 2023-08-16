from django.contrib import admin
from django.contrib.auth.models import User
from .models import Data, Agency
# Unregister the provided model admin
admin.site.unregister(User)


# Now you can register it again
admin.site.register(User)
admin.site.register(Data)
admin.site.register(Agency)
