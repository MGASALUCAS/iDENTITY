from django.db import models
from django.contrib.auth.models import User


# Create your models here.
class RegisteredStudents(models.Model):
    Name_Of_Student = models.CharField(max_length=255)  # this has used to show title / heading
    CollegeEnrolled = models.CharField(max_length=255)
    Student_Program = models.CharField(max_length=255)
    Date_Of_enrollment = models.DateTimeField()
    image = models.ImageField(upload_to='images/')
    image2 = models.ImageField(upload_to='images/')

    def __str__(self):
        return self.Name_Of_Student

    def pubdate_pretty(self):
        return self.Date_Of_enrollment.strftime('%b %e %Y')
