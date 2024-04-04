# ODS MLOps sample repo

Линтер и форматтирование осуществляется с поомщью ruff.

## Как запустить

Install [pdm](https://daobook.github.io/pdm/)

Установить зависимости:
```
pdm install --no-self --prod
```

Для разработки:
```
pdm install --no-self -d
```

## Работа с ветками

main - стабильная ветка, develop - экспериментальная ветка, feature - изменения.

```mermaid
gitGraph
    commit
    commit
    branch develop
    commit
    commit
    branch feature
    checkout feature
    commit
    commit
    checkout develop
    merge feature
    commit
    checkout main
    merge develop

```

