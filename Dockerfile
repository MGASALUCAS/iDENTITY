# Base Image
FROM python:3.9-alpine

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /code

# Install dependencies
RUN apk add --update --no-cache postgresql-client
RUN apk add --update --no-cache --virtual .tmp-build-deps \
      gcc libc-dev linux-headers postgresql-dev
COPY requirements.txt .
RUN pip install -U scikit-learn 
RUN pip install opencv-python
RUN pip install --no-cache-dir -r requirements.txt
RUN apk del .tmp-build-deps

# Copy project files to the container
COPY . .

# Run migrations
RUN python manage.py migrate

# Expose the port
EXPOSE 8000

# Start the server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
