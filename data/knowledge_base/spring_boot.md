# Spring Boot 使用说明

## 快速开始

Spring Boot 是一个用于快速构建独立 Spring 应用的框架，内嵌 Tomcat 服务器，
可以打包成单个可执行 JAR 直接运行。可以通过 start.spring.io 初始化项目，
选择依赖后生成工程。

## 配置文件

application.yml 是 Spring Boot 的默认配置文件，用于配置数据库连接、服务端口、
日志级别等。支持多环境配置，如 application-dev.yml 和 application-prod.yml。

## 自动装配

Spring Boot 的自动装配机制会根据 classpath 中的依赖自动配置 Bean，
减少手动配置的工作量。
