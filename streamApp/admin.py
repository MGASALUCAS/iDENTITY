from django.contrib import admin

# Register your models here.
from .models import RegisteredStudents

admin.site.register(RegisteredStudents)