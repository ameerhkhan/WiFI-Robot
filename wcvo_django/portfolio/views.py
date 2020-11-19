from django.shortcuts import render
from portfolio.models import Portfolio

# Create your views here.

def project_index(request):
    projects = Portfolio.objects.all()      # a query get all objects in the table/sheet.

    context = {                             # the context dictionary is used to send information to our
        'projects': projects                # templates. Every view function needs to have this.
    }

    return render(request, 'project_index.html', context)
    # render takes in the request, template and the context dictionary.

def project_detail(request):
    project = Portfolio.objects.get(pk=pk)  # Retrieve object with primary key (PK)

    context = {
        'project': project
    }

    return render(request, 'project_detail.html', context)