# Nginx ingress

Для зауска:
```
docker compose -f ./compose/docker-compose.web.yaml build
minikube start --driver=docker
minikube image load ods-mlops-web
```

Запустить приложение:
```
kubectl apply -f ./kubernetes/deploy.yaml
```

Для minikube (WSL 2):
```
minikube service web-server-service --url
minikube tunnel
```


Тестирование получения HTML страницы:
```
curl -v --resolve "ml.ods.io:80:127.0.0.1" -i http://ml.ods.io/
```

В других сценариях развёртывания необздимо использовать другой адрес вместо `127.0.0.1`.
