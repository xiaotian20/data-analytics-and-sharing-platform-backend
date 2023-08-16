from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, Birch
from django.contrib.auth import authenticate, login
from django.db.models import Q
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from sklearn.metrics import mean_squared_error, accuracy_score, silhouette_score
import numpy as np
from rest_framework.permissions import IsAuthenticated
from .models import Agency, Data
from .serializers import AgencySerializer, DataSerializer
import pandas as pd
import numpy as np
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import Agency, Data
from .serializers import AgencySerializer, DataSerializer
from diffprivlib.mechanisms import Laplace
from diffprivlib.tools.histograms import histogram
from django.core.files.base import ContentFile
from io import StringIO
from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate, login
from django.http import JsonResponse
from django.contrib.auth.forms import UserCreationForm
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from django.utils.decorators import method_decorator
import tempfile
import shutil
import os
@csrf_exempt
def register(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))  # Load JSON data from request body
        form = UserCreationForm(data)
        if form.is_valid():
            user = form.save()  # This will return the created User instance
            
            # Here, we create an Agency associated with the newly created User
            agency = Agency(user=user)
            agency.save()
            
            return JsonResponse({"status": "success"})
        else:
            return JsonResponse({"errors": form.errors}, status=400)
    else:
        return JsonResponse({"error": "Invalid method"}, status=405)

@csrf_exempt
def login_view(request):
    print(request.body)
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        username = data.get('username')
        password = data.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            agency_id = user.agency.id
            return JsonResponse({'status': 'Login success.', 'loginId': agency_id}, status=200)
        else:
            return JsonResponse({'status': 'Invalid username or password.'}, status=400)
    else:
        return JsonResponse({'status': 'Invalid request method.'}, status=400)




from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

@method_decorator(csrf_exempt, name='dispatch')
class AgencyViewset(viewsets.ModelViewSet):
    queryset = Agency.objects.all()
    serializer_class = AgencySerializer

    def get_queryset(self):
        agencyId = self.kwargs.get('agencyId', None)
        if agencyId is not None:
            return Agency.objects.filter(id=agencyId)
        else:
            return Agency.objects.all()

    @csrf_exempt
    @action(detail=True, methods=['post'], url_path='upload_data')
    def upload_data(self, request, pk=None):
        uploaded_file = request.FILES.get('file')  # Use FILES instead of data for file upload
        if not uploaded_file:
            return Response({'status': 'No data file provided.'}, status=400)
        loginId = self.kwargs['pk']
        agency = Agency.objects.get(id=loginId)
        public = request.data.get('public', False)
        if (public!=False):
            public = True
        privacy_cols = request.data.get('privacy_cols', None)
        epsilon = float(request.data.get('epsilon', 1.0))

        # Read uploaded file directly into a pandas DataFrame
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            return Response({'status': f'Error reading file: {str(e)}'}, status=400)
        if privacy_cols:
            privacy_cols = json.loads(privacy_cols)
            for col in privacy_cols:
                print(privacy_cols)
                if col in df.columns:
                    print("yes")
                    laplace = Laplace(epsilon=epsilon, sensitivity=1)
                    df[col] = df[col].apply(laplace.randomise)

        try:
            path = 'datasets/' + uploaded_file.name
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            path = default_storage.save(path, ContentFile(csv_buffer.getvalue()))
            new_data = Data(agency=agency, data=path, public=public)
            new_data.save()
        except Exception as e:
            error_message = str(e)
            return Response({'status': f'Error saving file: {error_message}'}, status=500)
        
        accessible_agencies = request.data.get('agencies', False)
        accessible_agencies = json.loads(accessible_agencies)
        print(accessible_agencies)
        if accessible_agencies:
            for agency_id in accessible_agencies:
                print(agency_id)
                try:
                    target_agency = Agency.objects.get(id=agency_id)
                    target_agency.accessible_data.add(new_data)
                    target_agency.save()
                except Agency.DoesNotExist:
                    return Response({'status': f'Agency with id {agency_id} does not exist'}, status=400)


        return Response({'status': 'Data uploaded.', 'new_data_id': new_data.pk})

class DataViewset(viewsets.ModelViewSet):
    queryset = Data.objects.all()
    serializer_class = DataSerializer

    @action(detail=True, methods=['post'])
    def fuse_data(self, request, pk=None):
        data = self.get_object()
        print(request.data)
        loginId = request.data.get("userId")
        print(loginId)
        agency = Agency.objects.get(id=loginId)
        if not data.public and agency not in data.agency.accessible_data.all():
            return Response({'status': 'Permission denied.'})
        print("yes")
        df1 = pd.read_csv(data.data.path)
        other_data_id = request.data.get('otherDataId')
        print(other_data_id)
        other_data = Data.objects.get(pk=other_data_id)
        if not other_data.public and agency not in other_data.agency.accessible_data.all():
            return Response({'status': 'Permission denied.'})
        print("yes")

        df2 = pd.read_csv(other_data.data.path)
        if request.data.get('type') == 'horizontal':
            df = pd.concat([df1, df2])
        else:  # vertical
            df = pd.merge(df1, df2, on=request.data.get('on'))


        new_data_name = request.data.get('description')
        print(new_data_name)
  
        new_data_public = request.data.get('public')
        print(new_data_public)

        try:
            print("yes")
            path = 'datasets/' + new_data_name + '.csv'
            print(path)
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            print(csv_buffer)
            path = default_storage.save(path, ContentFile(csv_buffer.getvalue()))
            print(path)
            new_data = Data(agency=agency, data=path, public=new_data_public)
            print(new_data)
            new_data.save()
        except Exception as e:
            error_message = str(e)
            return Response({'status': f'Error saving file: {error_message}'}, status=500)
        
        accessible_agencies = request.data.get('agencies', False)
        print(accessible_agencies)
        if accessible_agencies:
            for agency_id in accessible_agencies:
                print(agency_id)
                try:
                    target_agency = Agency.objects.get(id=agency_id)
                    target_agency.accessible_data.add(new_data)
                    target_agency.save()
                except Agency.DoesNotExist:
                    return Response({'status': f'Agency with id {agency_id} does not exist'}, status=400)



        new_data.save()

        return Response({'status': 'Data fused.', 'new_data_id': new_data.pk})




    
    def get_model_attributes(self, model, x, y, model_type):
        if model_type == 'clustering':
            model.fit(x)
            y_pred = model.labels_
        else:
            y_pred = model.predict(x)

        response = {}

        if model_type == 'regression':
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            response['mse'] = mse
            response['rmse'] = rmse

        if model_type == 'classification':
            acc = accuracy_score(y, y_pred)
            response['accuracy'] = acc
  
        if model_type == 'clustering':
            if len(np.unique(y_pred)) > 1:  
                silhouette = silhouette_score(x, y_pred)
                response['silhouette_score'] = silhouette
            else:  
                response['silhouette_score'] = 'Not applicable - only one cluster'

        if hasattr(model, 'coef_'):
            response['coef'] = model.coef_.tolist()

        if hasattr(model, 'intercept_'):
            response['intercept'] = model.intercept_.tolist()

        return Response(response)
    

    


    @action(detail=True, methods=['post'])
    def regression(self, request, pk=None):
        data = self.get_object()

        loginId = request.data.get("userId")
        print(loginId)
        agency = Agency.objects.get(id=loginId)
        if not data.public and agency not in data.agency.accessible_data.all():
            return Response({'Permission denied.'})
        df = pd.read_csv(data.data.path)

        model_name = request.data.get('model')
        print(model_name)
        input_features = request.data.get('input_features')
        print(input_features)
        output_feature = request.data.get('output_feature')
        print(output_feature)

        x = df[input_features]
        print(x)
        y = df[output_feature]
        print(y)

        if model_name == 'linear_regression':
            model = LinearRegression()
        elif model_name == 'decision_tree':
            model = DecisionTreeRegressor()
        elif model_name == 'random_forest':
            model = RandomForestRegressor()
        elif model_name == 'svr':
            model = SVR()
        elif model_name == 'k_neighbors':
            model = KNeighborsRegressor()
        elif model_name == 'mlp':
            model = MLPRegressor()
        model.fit(x, y)

        return self.get_model_attributes(model, x, y, "regression")




        
    @action(detail=True, methods=['post'])
    def classification(self, request, pk=None):
        data = self.get_object()
        loginId = request.data.get("userId")
        print(loginId)
        agency = Agency.objects.get(id=loginId)
        if not data.public and agency not in data.agency.accessible_data.all():
            return Response({'Permission denied.'})
        df = pd.read_csv(data.data.path)

        model_name = request.data.get('model')
        input_features = request.data.get('input_features')
        output_feature = request.data.get('output_feature')

        x = df[input_features]
        y = df[output_feature]

        if model_name == 'decision_tree':
            model = DecisionTreeClassifier()
        elif model_name == 'random_forest':
            model = RandomForestClassifier()
        elif model_name == 'k_neighbors':
            model = KNeighborsClassifier()
        elif model_name == 'mlp':
            model = MLPClassifier()

        model.fit(x, y)
        return self.get_model_attributes(model, x, y, "classification")

    @action(detail=True, methods=['post'])
    def clustering(self, request, pk=None):
        data = self.get_object()
        loginId = request.data.get("userId")
        print(loginId)
        agency = Agency.objects.get(id=loginId)
        if not data.public and agency not in data.agency.accessible_data.all():
            return Response({'Permission denied.'})
        df = pd.read_csv(data.data.path)

        model_name = request.data.get('model')
        input_features = request.data.get('input_features')

        x = df[input_features]

        if model_name == 'k_means':
            model = KMeans()
        elif model_name == 'dbscan':
            model = DBSCAN()
        elif model_name == 'agglomerative':
            model = AgglomerativeClustering()
        elif model_name == 'birch':
            model = Birch()
        model.fit(x)
        return self.get_model_attributes(model, x, None, "clustering")

"""
    @action(detail=True, methods=['post'])
    def approve_result(self, request, pk=None):
        # This is an example of a method for approving analysis results
        data = self.get_object()
        if not request.user.agency == data.agency:
            return Response({'status': 'Permission denied.'})

        approval_status = request.data.get('approval_status')
        if approval_status is None:
            return Response({'status': 'Approval status required.'})

        data.approved = approval_status
        data.save()

        return Response({'status': 'Result approved.'})
"""