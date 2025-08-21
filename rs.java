// Routing Service Main Application
package com.notification.platform.routing;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cache.annotation.EnableCaching;
import org.springframework.scheduling.annotation.EnableAsync;

@SpringBootApplication
@EnableCaching
@EnableAsync
public class RoutingServiceApplication {
    public static void main(String[] args) {
        SpringApplication.run(RoutingServiceApplication.class, args);
    }
}

// Domain Models
package com.notification.platform.routing.domain;

import lombok.Data;
import lombok.Builder;
import java.time.Instant;
import java.util.Map;
import java.util.Set;
import java.util.List;

@Data
@Builder
public class RoutingContext {
    private String notificationId;
    private String userId;
    private String capability;
    private Map<String, Object> payload;
    private Map<String, String> metadata;
    private Priority priority;
    private String correlationId;
    private Set<Route> routes;
    private UserPreferences userPreferences;
    private Instant timestamp;
}

@Data
@Builder
public class Route {
    private String routeId;
    private Channel channel;
    private String templateId;
    private boolean enabled;
    private Integer priority;
    private Map<String, String> configuration;
    private Set<String> supportedCapabilities;
}

@Data
@Builder
public class Channel {
    private String channelId;
    private String name;
    private ChannelType type;
    private boolean active;
    private Map<String, String> properties;
}

public enum ChannelType {
    MOBILE_IOS,
    MOBILE_ANDROID,
    WEB_PUSH,
    EMAIL,
    SMS,
    TEAMS,
    SLACK,
    IN_APP
}

@Data
@Builder
public class UserPreferences {
    private String userId;
    private Set<String> enabledChannels;
    private Set<String> disabledCapabilities;
    private Map<String, ChannelPreference> channelPreferences;
    private boolean optedOut;
    private Instant updatedAt;
}

@Data
@Builder
public class ChannelPreference {
    private String channelId;
    private boolean enabled;
    private QuietHours quietHours;
    private Integer dailyLimit;
    private Set<String> allowedCapabilities;
}

@Data
@Builder
public class QuietHours {
    private String startTime; // HH:mm format
    private String endTime;
    private String timezone;
}

@Data
@Builder
public class Template {
    private String templateId;
    private String name;
    private TemplateType type;
    private Map<String, String> content;
    private Map<String, Object> defaultValues;
    private boolean personalized;
    private String version;
}

public enum TemplateType {
    STATIC,
    DYNAMIC,
    PERSONALIZED
}

@Data
@Builder
public class RoutingDecision {
    private String notificationId;
    private List<ChannelMessage> channelMessages;
    private Instant timestamp;
    private String correlationId;
}

@Data
@Builder
public class ChannelMessage {
    private String messageId;
    private String notificationId;
    private ChannelType channel;
    private String templateId;
    private Map<String, Object> personalizedContent;
    private Map<String, String> metadata;
    private Priority priority;
    private String targetQueue;
}

public enum Priority {
    LOW, MEDIUM, HIGH, CRITICAL
}

// Services
package com.notification.platform.routing.service;

import com.notification.platform.routing.domain.*;
import com.notification.platform.routing.repository.*;
import com.notification.platform.routing.queue.*;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import java.util.*;
import java.util.stream.Collectors;
import java.time.Instant;
import java.time.LocalTime;
import java.time.ZoneId;
import java.time.ZonedDateTime;

@Slf4j
@Service
@RequiredArgsConstructor
public class RoutingService {
    
    private final CapabilityRouteMappingService capabilityService;
    private final UserPreferenceService preferenceService;
    private final TemplateService templateService;
    private final EventBridgePublisher eventBridgePublisher;
    private final MeterRegistry meterRegistry;
    
    public RoutingDecision routeNotification(RoutingContext context) {
        Timer.Sample sample = Timer.start(meterRegistry);
        
        try {
            log.info("Processing routing for notification: {}", context.getNotificationId());
            
            // Get routes for capability
            Set<Route> capabilityRoutes = capabilityService.getRoutesForCapability(context.getCapability());
            if (capabilityRoutes.isEmpty()) {
                log.warn("No routes found for capability: {}", context.getCapability());
                throw new NoRoutesFoundException("No routes configured for capability: " + context.getCapability());
            }
            
            // Get user preferences
            UserPreferences preferences = preferenceService.getUserPreferences(context.getUserId());
            
            // Apply routing rules
            List<Route> eligibleRoutes = applyRoutingRules(capabilityRoutes, preferences, context);
            
            // Build channel messages
            List<ChannelMessage> channelMessages = buildChannelMessages(eligibleRoutes, context);
            
            // Publish to EventBridge
            publishToEventBridge(channelMessages);
            
            RoutingDecision decision = RoutingDecision.builder()
                .notificationId(context.getNotificationId())
                .channelMessages(channelMessages)
                .timestamp(Instant.now())
                .correlationId(context.getCorrelationId())
                .build();
            
            log.info("Routing decision made for notification: {}, channels: {}", 
                context.getNotificationId(), 
                channelMessages.stream().map(m -> m.getChannel().toString()).collect(Collectors.joining(",")));
            
            return decision;
            
        } finally {
            sample.stop(Timer.builder("routing.processing.time")
                .tag("capability", context.getCapability())
                .register(meterRegistry));
        }
    }
    
    private List<Route> applyRoutingRules(Set<Route> capabilityRoutes, 
                                          UserPreferences preferences, 
                                          RoutingContext context) {
        
        // Filter routes based on user preferences
        return capabilityRoutes.stream()
            .filter(route -> isRouteEnabled(route, preferences))
            .filter(route -> isChannelEnabled(route.getChannel(), preferences))
            .filter(route -> !isInQuietHours(route.getChannel(), preferences))
            .filter(route -> !isDailyLimitExceeded(route.getChannel(), preferences, context.getUserId()))
            .sorted(Comparator.comparing(Route::getPriority))
            .collect(Collectors.toList());
    }
    
    private boolean isRouteEnabled(Route route, UserPreferences preferences) {
        if (!route.isEnabled()) {
            return false;
        }
        
        if (preferences.isOptedOut()) {
            log.debug("User has opted out of all notifications");
            return false;
        }
        
        return true;
    }
    
    private boolean isChannelEnabled(Channel channel, UserPreferences preferences) {
        if (!channel.isActive()) {
            return false;
        }
        
        if (preferences.getEnabledChannels() != null && 
            !preferences.getEnabledChannels().contains(channel.getChannelId())) {
            return false;
        }
        
        var channelPref = preferences.getChannelPreferences().get(channel.getChannelId());
        if (channelPref != null && !channelPref.isEnabled()) {
            return false;
        }
        
        return true;
    }
    
    private boolean isInQuietHours(Channel channel, UserPreferences preferences) {
        var channelPref = preferences.getChannelPreferences().get(channel.getChannelId());
        if (channelPref == null || channelPref.getQuietHours() == null) {
            return false;
        }
        
        QuietHours quietHours = channelPref.getQuietHours();
        ZoneId zoneId = ZoneId.of(quietHours.getTimezone());
        ZonedDateTime now = ZonedDateTime.now(zoneId);
        LocalTime currentTime = now.toLocalTime();
        
        LocalTime startTime = LocalTime.parse(quietHours.getStartTime());
        LocalTime endTime = LocalTime.parse(quietHours.getEndTime());
        
        if (startTime.isBefore(endTime)) {
            return currentTime.isAfter(startTime) && currentTime.isBefore(endTime);
        } else {
            // Quiet hours span midnight
            return currentTime.isAfter(startTime) || currentTime.isBefore(endTime);
        }
    }
    
    private boolean isDailyLimitExceeded(Channel channel, UserPreferences preferences, String userId) {
        var channelPref = preferences.getChannelPreferences().get(channel.getChannelId());
        if (channelPref == null || channelPref.getDailyLimit() == null) {
            return false;
        }
        
        // Check daily count from cache/database
        int dailyCount = getDailyNotificationCount(userId, channel.getChannelId());
        return dailyCount >= channelPref.getDailyLimit();
    }
    
    private List<ChannelMessage> buildChannelMessages(List<Route> routes, RoutingContext context) {
        List<ChannelMessage> messages = new ArrayList<>();
        
        for (Route route : routes) {
            try {
                Template template = templateService.getTemplate(route.getTemplateId());
                Map<String, Object> personalizedContent = personalizeTemplate(template, context);
                
                ChannelMessage message = ChannelMessage.builder()
                    .messageId(UUID.randomUUID().toString())
                    .notificationId(context.getNotificationId())
                    .channel(route.getChannel().getType())
                    .templateId(template.getTemplateId())
                    .personalizedContent(personalizedContent)
                    .metadata(context.getMetadata())
                    .priority(context.getPriority())
                    .targetQueue(getTargetQueue(route.getChannel().getType()))
                    .build();
                    
                messages.add(message);
                
            } catch (Exception e) {
                log.error("Failed to build channel message for route: {}", route.getRouteId(), e);
            }
        }
        
        return messages;
    }
    
    private Map<String, Object> personalizeTemplate(Template template, RoutingContext context) {
        if (template.getType() == TemplateType.STATIC) {
            return Map.of("content", template.getContent());
        }
        
        // For dynamic templates, merge payload with template
        Map<String, Object> personalized = new HashMap<>(template.getDefaultValues());
        personalized.putAll(context.getPayload());
        
        if (template.isPersonalized()) {
            // Add user-specific personalization
            var userProfile = getUserProfile(context.getUserId());
            personalized.put("userName", userProfile.getName());
            personalized.put("userEmail", userProfile.getEmail());
            // Add more personalization as needed
        }
        
        return personalized;
    }
    
    private String getTargetQueue(ChannelType channelType) {
        return switch (channelType) {
            case MOBILE_IOS, MOBILE_ANDROID -> "mobile-queue";
            case WEB_PUSH -> "web-queue";
            case EMAIL -> "email-queue";
            case SMS -> "sms-queue";
            case TEAMS -> "teams-queue";
            case SLACK -> "slack-queue";
            case IN_APP -> "in-app-queue";
        };
    }
    
    private void publishToEventBridge(List<ChannelMessage> messages) {
        messages.forEach(message -> {
            try {
                eventBridgePublisher.publish(message);
                log.debug("Published message to EventBridge: {}", message.getMessageId());
            } catch (Exception e) {
                log.error("Failed to publish to EventBridge", e);
                throw new EventBridgePublishException("Failed to publish message", e);
            }
        });
    }
    
    private int getDailyNotificationCount(String userId, String channelId) {
        // Implementation would query from cache/database
        return 0;
    }
    
    private UserProfile getUserProfile(String userId) {
        // Implementation would fetch from user service
        return UserProfile.builder()
            .userId(userId)
            .name("User Name")
            .email("user@example.com")
            .build();
    }
}

// SQS Message Listener
package com.notification.platform.routing.queue;

import com.notification.platform.routing.domain.RoutingContext;
import com.notification.platform.routing.service.RoutingService;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import software.amazon.awssdk.services.sqs.SqsClient;
import software.amazon.awssdk.services.sqs.model.*;
import org.springframework.scheduling.annotation.Scheduled;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.springframework.beans.factory.annotation.Value;

@Slf4j
@Component
@RequiredArgsConstructor
public class SQSMessageListener {
    
    private final SqsClient sqsClient;
    private final ObjectMapper objectMapper;
    private final RoutingService routingService;
    private final ExecutorService executorService = Executors.newFixedThreadPool(10);
    
    @Value("${aws.sqs.routing-queue-url}")
    private String queueUrl;
    
    @Value("${aws.sqs.max-messages:10}")
    private int maxMessages;
    
    @Value("${aws.sqs.visibility-timeout:30}")
    private int visibilityTimeout;
    
    @Scheduled(fixedDelay = 1000) // Poll every second
    public void pollMessages() {
        try {
            ReceiveMessageRequest request = ReceiveMessageRequest.builder()
                .queueUrl(queueUrl)
                .maxNumberOfMessages(maxMessages)
                .visibilityTimeout(visibilityTimeout)
                .waitTimeSeconds(20) // Long polling
                .messageAttributeNames("All")
                .build();
                
            ReceiveMessageResponse response = sqsClient.receiveMessage(request);
            List<Message> messages = response.messages();
            
            if (!messages.isEmpty()) {
                log.info("Received {} messages from SQS", messages.size());
                processMessages(messages);
            }
            
        } catch (Exception e) {
            log.error("Error polling SQS messages", e);
        }
    }
    
    private void processMessages(List<Message> messages) {
        List<CompletableFuture<Void>> futures = messages.stream()
            .map(message -> CompletableFuture.runAsync(() -> processMessage(message), executorService))
            .toList();
            
        CompletableFuture.allOf(futures.toArray(new CompletableFuture[0]))
            .join();
    }
    
    private void processMessage(Message message) {
        try {
            log.debug("Processing message: {}", message.messageId());
            
            RoutingContext context = objectMapper.readValue(message.body(), RoutingContext.class);
            
            // Process routing
            routingService.routeNotification(context);
            
            // Delete message after successful processing
            deleteMessage(message.receiptHandle());
            
        } catch (Exception e) {
            log.error("Failed to process message: {}", message.messageId(), e);
            // Message will become visible again after visibility timeout
        }
    }
    
    private void deleteMessage(String receiptHandle) {
        try {
            DeleteMessageRequest deleteRequest = DeleteMessageRequest.builder()
                .queueUrl(queueUrl)
                .receiptHandle(receiptHandle)
                .build();
                
            sqsClient.deleteMessage(deleteRequest);
            log.debug("Deleted message with receipt handle: {}", receiptHandle);
            
        } catch (Exception e) {
            log.error("Failed to delete message", e);
        }
    }
}

// EventBridge Publisher
package com.notification.platform.routing.queue;

import com.notification.platform.routing.domain.ChannelMessage;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.retry.annotation.Backoff;
import org.springframework.retry.annotation.Retryable;
import org.springframework.stereotype.Component;
import software.amazon.awssdk.services.eventbridge.EventBridgeClient;
import software.amazon.awssdk.services.eventbridge.model.*;
import java.time.Instant;
import java.util.List;

@Slf4j
@Component
@RequiredArgsConstructor
public class EventBridgePublisher {
    
    private final EventBridgeClient eventBridgeClient;
    private final ObjectMapper objectMapper;
    
    @Value("${aws.eventbridge.bus-name}")
    private String eventBusName;
    
    @Value("${aws.eventbridge.source}")
    private String eventSource;
    
    @Retryable(value = Exception.class, maxAttempts = 3, backoff = @Backoff(delay = 1000))
    public void publish(ChannelMessage message) {
        try {
            String detailJson = objectMapper.writeValueAsString(message);
            
            PutEventsRequestEntry entry = PutEventsRequestEntry.builder()
                .eventBusName(eventBusName)
                .source(eventSource)
                .detailType(getDetailType(message.getChannel()))
                .detail(detailJson)
                .time(Instant.now())
                .build();
                
            PutEventsRequest request = PutEventsRequest.builder()
                .entries(List.of(entry))
                .build();
                
            PutEventsResponse response = eventBridgeClient.putEvents(request);
            
            if (response.failedEntryCount() > 0) {
                log.error("Failed to publish events: {}", response.entries());
                throw new EventBridgePublishException("Failed to publish event to EventBridge");
            }
            
            log.info("Published message to EventBridge: {}", message.getMessageId());
            
        } catch (Exception e) {
            log.error("Error publishing to EventBridge", e);
            throw new EventBridgePublishException("Failed to publish to EventBridge", e);
        }
    }
    
    private String getDetailType(ChannelType channel) {
        return "notification." + channel.name().toLowerCase();
    }
}

// Configuration
package com.notification.platform.routing.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import software.amazon.awssdk.auth.credentials.DefaultCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.sqs.SqsClient;
import software.amazon.awssdk.services.eventbridge.EventBridgeClient;
import org.springframework.cache.CacheManager;
import org.springframework.cache.caffeine.CaffeineCacheManager;
import com.github.benmanes.caffeine.cache.Caffeine;
import java.util.concurrent.TimeUnit;

@Configuration
public class AwsConfig {
    
    @Value("${aws.region}")
    private String awsRegion;
    
    @Bean
    public SqsClient sqsClient() {
        return SqsClient.builder()
            .region(Region.of(awsRegion))
            .credentialsProvider(DefaultCredentialsProvider.create())
            .build();
    }
    
    @Bean
    public EventBridgeClient eventBridgeClient() {
        return EventBridgeClient.builder()
            .region(Region.of(awsRegion))
            .credentialsProvider(DefaultCredentialsProvider.create())
            .build();
    }
    
    @Bean
    public CacheManager cacheManager() {
        CaffeineCacheManager cacheManager = new CaffeineCacheManager();
        cacheManager.setCaffeine(Caffeine.newBuilder()
            .maximumSize(10000)
            .expireAfterWrite(5, TimeUnit.MINUTES));
        return cacheManager;
    }
}

// Application Properties
# application.yml
spring:
  application:
    name: routing-service
  data:
    mongodb:
      uri: ${MONGODB_URI:mongodb://localhost:27017/routing}
  cache:
    type: caffeine

aws:
  region: ${AWS_REGION:us-east-1}
  sqs:
    routing-queue-url: ${ROUTING_QUEUE_URL}
    max-messages: 10
    visibility-timeout: 30
  eventbridge:
    bus-name: ${EVENT_BUS_NAME:notification-bus}
    source: routing-service

management:
  endpoints:
    web:
      exposure:
        include: health,info,metrics,prometheus
  metrics:
    export:
      prometheus:
        enabled: true

logging:
  level:
    com.notification.platform: DEBUG
