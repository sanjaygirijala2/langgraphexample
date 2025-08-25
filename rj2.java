# application.yml
spring:
  application:
    name: router-service
  
  profiles:
    active: ${SPRING_PROFILES_ACTIVE:dev}
  
  data:
    mongodb:
      uri: ${MONGODB_URI:mongodb://localhost:27017/notification-platform}
      database: ${MONGODB_DATABASE:notification-platform}
      auto-index-creation: true
  
  cache:
    type: caffeine
    caffeine:
      spec: maximumSize=10000,expireAfterWrite=5m
  
  jackson:
    serialization:
      write-dates-as-timestamps: false
    deserialization:
      fail-on-unknown-properties: false

aws:
  region: ${AWS_REGION:us-east-1}
  sqs:
    router-queue: ${ROUTER_QUEUE_NAME:notification-router-queue}
    max-messages: 10
    visibility-timeout: 30
    wait-time-seconds: 20
  eventbridge:
    bus-name: ${EVENT_BUS_NAME:notification-event-bus}
    source: notification-router

notification:
  quiet-hours:
    enabled: ${QUIET_HOURS_ENABLED:true}
    start: ${QUIET_HOURS_START:22}
    end: ${QUIET_HOURS_END:8}
  
  batch:
    size: ${BATCH_SIZE:100}
    thread-pool-size: ${THREAD_POOL_SIZE:10}

resilience4j:
  circuitbreaker:
    instances:
      eventbridge:
        registerHealthIndicator: true
        slidingWindowSize: 10
        minimumNumberOfCalls: 5
        failureRateThreshold: 50
        waitDurationInOpenState: 30s
        permittedNumberOfCallsInHalfOpenState: 3
        automaticTransitionFromOpenToHalfOpenEnabled: true
  
  retry:
    instances:
      eventbridge:
        maxAttempts: 3
        waitDuration: 1s
        enableExponentialBackoff: true
        exponentialBackoffMultiplier: 2

management:
  endpoints:
    web:
      exposure:
        include: health,info,metrics,prometheus
  metrics:
    export:
      prometheus:
        enabled: true
    tags:
      application: ${spring.application.name}
      environment: ${spring.profiles.active}

logging:
  level:
    root: INFO
    com.notification.platform.router: DEBUG
    org.springframework.data.mongodb: DEBUG
  pattern:
    console: "%d{yyyy-MM-dd HH:mm:ss} - %logger{36} - %msg%n"
    file: "%d{yyyy-MM-dd HH:mm:ss} [%thread] %-5level %logger{36} - %msg%n"
  file:
    name: logs/router-service.log
    max-size: 10MB
    max-history: 30

---
# application-dev.yml
spring:
  config:
    activate:
      on-profile: dev
  
  data:
    mongodb:
      uri: mongodb://localhost:27017/notification-platform-dev

aws:
  sqs:
    router-queue: notification-router-queue-dev
  eventbridge:
    bus-name: notification-event-bus-dev

logging:
  level:
    com.notification.platform.router: DEBUG
    org.springframework: DEBUG

---
# application-prod.yml
spring:
  config:
    activate:
      on-profile: prod
  
  data:
    mongodb:
      uri: ${MONGODB_URI}
      ssl: true
      connection-timeout: 10000
      socket-timeout: 10000
      max-connection-idle-time: 60000

aws:
  sqs:
    router-queue: ${ROUTER_QUEUE_NAME}
    max-messages: 25
    visibility-timeout: 60
  eventbridge:
    bus-name: ${EVENT_BUS_NAME}

notification:
  batch:
    size: 500
    thread-pool-size: 20

logging:
  level:
    root: WARN
    com.notification.platform.router: INFO
  file:
    name: /var/log/router-service/router-service.log

management:
  endpoints:
    web:
      exposure:
        include: health,metrics,prometheus
