// Routing Service Main Application
package com.notification.platform.routing;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cache.annotation.EnableCaching;
import org.springframework.scheduling.annotation.EnableAsync;
import org.springframework.scheduling.annotation.EnableScheduling;

@SpringBootApplication
@EnableCaching
@EnableAsync
@EnableScheduling
public class RoutingServiceApplication {
    public static void main(String[] args) {
        SpringApplication.run(RoutingServiceApplication.class, args);
    }
}

// Enhanced Domain Models
package com.notification.platform.routing.domain;

import lombok.Data;
import lombok.Builder;
import java.time.Instant;
import java.time.ZonedDateTime;
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
    private boolean smartNotification;
    private String userTimeZone;
    private ZonedDateTime scheduledDeliveryTime;
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
    private String timezone;
    private PreferredDeliveryTime preferredDeliveryTime;
}

@Data
@Builder
public class PreferredDeliveryTime {
    private String startTime; // HH:mm format
    private String endTime;
    private Set<String> preferredDays; // MON, TUE, WED, etc.
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
public class UserPresence {
    private String userId;
    private Map<ChannelType, PresenceStatus> channelPresence;
    private ChannelType mostRecentActiveChannel;
    private Instant lastActivityTime;
    private Map<ChannelType, Instant> lastSeenTimestamps;
}

@Data
@Builder
public class PresenceStatus {
    private boolean isOnline;
    private boolean isActive;
    private Instant lastSeen;
    private String deviceId;
    private String sessionId;
}

@Data
@Builder
public class EmployeeData {
    private String employeeId;
    private String firstName;
    private String lastName;
    private String email;
    private String department;
    private String title;
    private String managerId;
    private String location;
    private String timezone;
    private Map<String, String> additionalAttributes;
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
    private List<DeepLinkPlaceholder> deepLinkPlaceholders;
}

@Data
@Builder
public class DeepLinkPlaceholder {
    private String placeholder;
    private String actionType;
    private Map<String, String> parameters;
}

@Data
@Builder
public class DeepLink {
    private String actionType;
    private String url;
    private String webUrl;
    private String mobileUrl;
    private Map<String, String> parameters;
    private String displayText;
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
    private ZonedDateTime scheduledDeliveryTime;
    private boolean isSmartNotification;
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
    private ZonedDateTime scheduledDeliveryTime;
    private List<DeepLink> deepLinks;
}

public enum Priority {
    LOW, MEDIUM, HIGH, CRITICAL
}

// Enhanced Routing Service
package com.notification.platform.routing.service;

import com.notification.platform.routing.domain.*;
import com.notification.platform.routing.repository.*;
import com.notification.platform.routing.queue.*;
import com.notification.platform.routing.external.*;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import java.util.*;
import java.util.stream.Collectors;
import java.time.*;
import java.time.temporal.ChronoUnit;
import org.springframework.cache.annotation.Cacheable;

@Slf4j
@Service
@RequiredArgsConstructor
public class RoutingService {
    
    private final CapabilityRouteMappingService capabilityService;
    private final UserPreferenceService preferenceService;
    private final TemplateService templateService;
    private final EventBridgePublisher eventBridgePublisher;
    private final UserPresenceService presenceService;
    private final EmployeeDataService employeeDataService;
    private final BFFService bffService;
    private final SmartSchedulingService smartSchedulingService;
    private final MeterRegistry meterRegistry;
    
    public RoutingDecision routeNotification(RoutingContext context) {
        Timer.Sample sample = Timer.start(meterRegistry);
        
        try {
            log.info("Processing routing for notification: {}, smartNotification: {}", 
                context.getNotificationId(), context.isSmartNotification());
            
            // Get routes for capability
            Set<Route> capabilityRoutes = capabilityService.getRoutesForCapability(context.getCapability());
            if (capabilityRoutes.isEmpty()) {
                log.warn("No routes found for capability: {}", context.getCapability());
                throw new NoRoutesFoundException("No routes configured for capability: " + context.getCapability());
            }
            
            // Get user preferences
            UserPreferences preferences = preferenceService.getUserPreferences(context.getUserId());
            
            // Get user presence information
            UserPresence userPresence = presenceService.getUserPresence(context.getUserId());
            
            // Handle smart notification scheduling
            ZonedDateTime scheduledTime = null;
            if (context.isSmartNotification()) {
                scheduledTime = smartSchedulingService.calculateOptimalDeliveryTime(
                    context.getUserId(), 
                    preferences, 
                    userPresence,
                    context.getPriority()
                );
                
                if (scheduledTime != null && scheduledTime.isAfter(ZonedDateTime.now())) {
                    log.info("Smart notification scheduled for: {} for user: {}", 
                        scheduledTime, context.getUserId());
                    context.setScheduledDeliveryTime(scheduledTime);
                }
            }
            
            // Apply routing rules with presence awareness
            List<Route> eligibleRoutes = applyRoutingRules(
                capabilityRoutes, 
                preferences, 
                userPresence, 
                context
            );
            
            // Get employee data for personalization
            EmployeeData employeeData = fetchEmployeeData(context.getUserId());
            
            // Build channel messages with enrichment
            List<ChannelMessage> channelMessages = buildEnrichedChannelMessages(
                eligibleRoutes, 
                context, 
                employeeData
            );
            
            // If smart notification and scheduled for later, store for delayed processing
            if (scheduledTime != null && scheduledTime.isAfter(ZonedDateTime.now())) {
                scheduleDelayedDelivery(channelMessages, scheduledTime);
            } else {
                // Publish to EventBridge immediately
                publishToEventBridge(channelMessages);
            }
            
            RoutingDecision decision = RoutingDecision.builder()
                .notificationId(context.getNotificationId())
                .channelMessages(channelMessages)
                .timestamp(Instant.now())
                .correlationId(context.getCorrelationId())
                .scheduledDeliveryTime(scheduledTime)
                .isSmartNotification(context.isSmartNotification())
                .build();
            
            log.info("Routing decision made for notification: {}, channels: {}", 
                context.getNotificationId(), 
                channelMessages.stream().map(m -> m.getChannel().toString()).collect(Collectors.joining(",")));
            
            return decision;
            
        } finally {
            sample.stop(Timer.builder("routing.processing.time")
                .tag("capability", context.getCapability())
                .tag("smart", String.valueOf(context.isSmartNotification()))
                .register(meterRegistry));
        }
    }
    
    private List<Route> applyRoutingRules(Set<Route> capabilityRoutes, 
                                          UserPreferences preferences,
                                          UserPresence presence,
                                          RoutingContext context) {
        
        // Get active channels based on presence
        Set<ChannelType> activeChannels = getActiveChannels(presence);
        
        // Filter routes based on user preferences and presence
        List<Route> eligibleRoutes = capabilityRoutes.stream()
            .filter(route -> isRouteEnabled(route, preferences))
            .filter(route -> isChannelEnabled(route.getChannel(), preferences))
            .filter(route -> !isInQuietHours(route.getChannel(), preferences))
            .filter(route -> !isDailyLimitExceeded(route.getChannel(), preferences, context.getUserId()))
            .sorted((r1, r2) -> {
                // Prioritize active channels
                boolean r1Active = activeChannels.contains(r1.getChannel().getType());
                boolean r2Active = activeChannels.contains(r2.getChannel().getType());
                
                if (r1Active && !r2Active) return -1;
                if (!r1Active && r2Active) return 1;
                
                // Then sort by configured priority
                return r1.getPriority().compareTo(r2.getPriority());
            })
            .collect(Collectors.toList());
        
        // If user is active on a specific channel and it's eligible, prioritize it
        if (presence.getMostRecentActiveChannel() != null) {
            eligibleRoutes = prioritizeActiveChannel(eligibleRoutes, presence.getMostRecentActiveChannel());
        }
        
        return eligibleRoutes;
    }
    
    private Set<ChannelType> getActiveChannels(UserPresence presence) {
        if (presence == null || presence.getChannelPresence() == null) {
            return new HashSet<>();
        }
        
        return presence.getChannelPresence().entrySet().stream()
            .filter(entry -> entry.getValue().isOnline() || entry.getValue().isActive())
            .map(Map.Entry::getKey)
            .collect(Collectors.toSet());
    }
    
    private List<Route> prioritizeActiveChannel(List<Route> routes, ChannelType activeChannel) {
        // Move the active channel route to the front if it exists
        return routes.stream()
            .sorted((r1, r2) -> {
                if (r1.getChannel().getType() == activeChannel) return -1;
                if (r2.getChannel().getType() == activeChannel) return 1;
                return 0;
            })
            .collect(Collectors.toList());
    }
    
    @Cacheable(value = "employeeData", key = "#userId")
    private EmployeeData fetchEmployeeData(String userId) {
        try {
            return employeeDataService.getEmployeeData(userId);
        } catch (Exception e) {
            log.error("Failed to fetch employee data for user: {}", userId, e);
            // Return minimal data if service fails
            return EmployeeData.builder()
                .employeeId(userId)
                .firstName("User")
                .lastName("")
                .build();
        }
    }
    
    private List<ChannelMessage> buildEnrichedChannelMessages(List<Route> routes, 
                                                              RoutingContext context,
                                                              EmployeeData employeeData) {
        List<ChannelMessage> messages = new ArrayList<>();
        
        for (Route route : routes) {
            try {
                Template template = templateService.getTemplate(route.getTemplateId());
                
                // Personalize template with employee data
                Map<String, Object> personalizedContent = personalizeTemplate(
                    template, 
                    context, 
                    employeeData
                );
                
                // Generate deep links
                List<DeepLink> deepLinks = generateDeepLinks(
                    template, 
                    context, 
                    route.getChannel().getType()
                );
                
                ChannelMessage message = ChannelMessage.builder()
                    .messageId(UUID.randomUUID().toString())
                    .notificationId(context.getNotificationId())
                    .channel(route.getChannel().getType())
                    .templateId(template.getTemplateId())
                    .personalizedContent(personalizedContent)
                    .metadata(context.getMetadata())
                    .priority(context.getPriority())
                    .targetQueue(getTargetQueue(route.getChannel().getType()))
                    .scheduledDeliveryTime(context.getScheduledDeliveryTime())
                    .deepLinks(deepLinks)
                    .build();
                    
                messages.add(message);
                
            } catch (Exception e) {
                log.error("Failed to build channel message for route: {}", route.getRouteId(), e);
            }
        }
        
        return messages;
    }
    
    private Map<String, Object> personalizeTemplate(Template template, 
                                                   RoutingContext context,
                                                   EmployeeData employeeData) {
        Map<String, Object> personalized = new HashMap<>();
        
        // Add default values
        if (template.getDefaultValues() != null) {
            personalized.putAll(template.getDefaultValues());
        }
        
        // Add payload data
        if (context.getPayload() != null) {
            personalized.putAll(context.getPayload());
        }
        
        // Add employee personalization
        if (employeeData != null) {
            personalized.put("firstName", employeeData.getFirstName());
            personalized.put("lastName", employeeData.getLastName());
            personalized.put("fullName", employeeData.getFirstName() + " " + employeeData.getLastName());
            personalized.put("email", employeeData.getEmail());
            personalized.put("department", employeeData.getDepartment());
            personalized.put("title", employeeData.getTitle());
            personalized.put("location", employeeData.getLocation());
            
            // Add any additional attributes
            if (employeeData.getAdditionalAttributes() != null) {
                personalized.putAll(employeeData.getAdditionalAttributes());
            }
        }
        
        // Process template content with personalized data
        if (template.getContent() != null) {
            Map<String, String> processedContent = new HashMap<>();
            for (Map.Entry<String, String> entry : template.getContent().entrySet()) {
                processedContent.put(entry.getKey(), 
                    processTemplate(entry.getValue(), personalized));
            }
            personalized.put("content", processedContent);
        }
        
        return personalized;
    }
    
    private String processTemplate(String template, Map<String, Object> data) {
        String processed = template;
        for (Map.Entry<String, Object> entry : data.entrySet()) {
            String placeholder = "{{" + entry.getKey() + "}}";
            if (entry.getValue() != null) {
                processed = processed.replace(placeholder, entry.getValue().toString());
            }
        }
        return processed;
    }
    
    private List<DeepLink> generateDeepLinks(Template template, 
                                            RoutingContext context,
                                            ChannelType channelType) {
        List<DeepLink> deepLinks = new ArrayList<>();
        
        if (template.getDeepLinkPlaceholders() == null || template.getDeepLinkPlaceholders().isEmpty()) {
            return deepLinks;
        }
        
        for (DeepLinkPlaceholder placeholder : template.getDeepLinkPlaceholders()) {
            try {
                DeepLink deepLink = bffService.generateDeepLink(
                    placeholder.getActionType(),
                    placeholder.getParameters(),
                    context.getUserId(),
                    channelType
                );
                
                if (deepLink != null) {
                    deepLinks.add(deepLink);
                }
            } catch (Exception e) {
                log.error("Failed to generate deep link for action: {}", 
                    placeholder.getActionType(), e);
            }
        }
        
        return deepLinks;
    }
    
    private void scheduleDelayedDelivery(List<ChannelMessage> messages, ZonedDateTime scheduledTime) {
        // Store messages for delayed delivery
        messages.forEach(message -> {
            smartSchedulingService.scheduleMessage(message, scheduledTime);
        });
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
}

// External Service Clients
package com.notification.platform.routing.external;

import com.notification.platform.routing.domain.*;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.cloud.circuitbreaker.resilience4j.Resilience4JCircuitBreakerFactory;
import org.springframework.cloud.client.circuitbreaker.CircuitBreaker;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.*;
import lombok.RequiredArgsConstructor;
import java.util.Map;
import java.util.HashMap;

@Slf4j
@Service
@RequiredArgsConstructor
public class UserPresenceService {
    
    private final RestTemplate restTemplate;
    private final Resilience4JCircuitBreakerFactory circuitBreakerFactory;
    
    @Value("${external.services.user-presence.url}")
    private String userPresenceServiceUrl;
    
    @Cacheable(value = "userPresence", key = "#userId", unless = "#result == null")
    public UserPresence getUserPresence(String userId) {
        CircuitBreaker circuitBreaker = circuitBreakerFactory.create("user-presence");
        
        return circuitBreaker.run(
            () -> fetchUserPresence(userId),
            throwable -> {
                log.error("Error fetching user presence for user: {}", userId, throwable);
                return getDefaultUserPresence(userId);
            }
        );
    }
    
    private UserPresence fetchUserPresence(String userId) {
        String url = userPresenceServiceUrl + "/api/v1/presence/" + userId;
        
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        
        HttpEntity<String> entity = new HttpEntity<>(headers);
        
        ResponseEntity<UserPresence> response = restTemplate.exchange(
            url,
            HttpMethod.GET,
            entity,
            UserPresence.class
        );
        
        return response.getBody();
    }
    
    private UserPresence getDefaultUserPresence(String userId) {
        // Return default presence when service is unavailable
        return UserPresence.builder()
            .userId(userId)
            .channelPresence(new HashMap<>())
            .build();
    }
}

@Slf4j
@Service
@RequiredArgsConstructor
public class EmployeeDataService {
    
    private final RestTemplate restTemplate;
    private final Resilience4JCircuitBreakerFactory circuitBreakerFactory;
    
    @Value("${external.services.employee-data.url}")
    private String employeeDataServiceUrl;
    
    @Cacheable(value = "employeeData", key = "#employeeId", unless = "#result == null")
    public EmployeeData getEmployeeData(String employeeId) {
        CircuitBreaker circuitBreaker = circuitBreakerFactory.create("employee-data");
        
        return circuitBreaker.run(
            () -> fetchEmployeeData(employeeId),
            throwable -> {
                log.error("Error fetching employee data for: {}", employeeId, throwable);
                return getDefaultEmployeeData(employeeId);
            }
        );
    }
    
    private EmployeeData fetchEmployeeData(String employeeId) {
        String url = employeeDataServiceUrl + "/api/v1/employees/" + employeeId;
        
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        
        HttpEntity<String> entity = new HttpEntity<>(headers);
        
        ResponseEntity<EmployeeData> response = restTemplate.exchange(
            url,
            HttpMethod.GET,
            entity,
            EmployeeData.class
        );
        
        return response.getBody();
    }
    
    private EmployeeData getDefaultEmployeeData(String employeeId) {
        return EmployeeData.builder()
            .employeeId(employeeId)
            .firstName("User")
            .lastName("")
            .build();
    }
}

@Slf4j
@Service
@RequiredArgsConstructor
public class BFFService {
    
    private final RestTemplate restTemplate;
    private final Resilience4JCircuitBreakerFactory circuitBreakerFactory;
    
    @Value("${external.services.bff.url}")
    private String bffServiceUrl;
    
    public DeepLink generateDeepLink(String actionType, 
                                    Map<String, String> parameters,
                                    String userId,
                                    ChannelType channelType) {
        CircuitBreaker circuitBreaker = circuitBreakerFactory.create("bff-service");
        
        return circuitBreaker.run(
            () -> fetchDeepLink(actionType, parameters, userId, channelType),
            throwable -> {
                log.error("Error generating deep link for action: {}", actionType, throwable);
                return getDefaultDeepLink(actionType, parameters);
            }
        );
    }
    
    private DeepLink fetchDeepLink(String actionType,
                                  Map<String, String> parameters,
                                  String userId,
                                  ChannelType channelType) {
        String url = bffServiceUrl + "/api/v1/deeplinks/generate";
        
        Map<String, Object> request = new HashMap<>();
        request.put("actionType", actionType);
        request.put("parameters", parameters);
        request.put("userId", userId);
        request.put("channelType", channelType.toString());
        
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        
        HttpEntity<Map<String, Object>> entity = new HttpEntity<>(request, headers);
        
        ResponseEntity<DeepLink> response = restTemplate.exchange(
            url,
            HttpMethod.POST,
            entity,
            DeepLink.class
        );
        
        return response.getBody();
    }
    
    private DeepLink getDefaultDeepLink(String actionType, Map<String, String> parameters) {
        return DeepLink.builder()
            .actionType(actionType)
            .url("#")
            .parameters(parameters)
            .displayText("View Details")
            .build();
    }
}

// Smart Scheduling Service
package com.notification.platform.routing.service;

import com.notification.platform.routing.domain.*;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.data.mongodb.core.query.Criteria;
import org.springframework.data.mongodb.core.query.Query;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import java.time.*;
import java.time.temporal.ChronoUnit;
import java.util.*;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
public class SmartSchedulingService {
    
    private final MongoTemplate mongoTemplate;
    private final EventBridgePublisher eventBridgePublisher;
    
    public ZonedDateTime calculateOptimalDeliveryTime(String userId,
                                                     UserPreferences preferences,
                                                     UserPresence presence,
                                                     Priority priority) {
        
        // For critical priority, deliver immediately
        if (priority == Priority.CRITICAL) {
            return ZonedDateTime.now();
        }
        
        String timezone = preferences.getTimezone() != null ? 
            preferences.getTimezone() : "UTC";
        ZoneId zoneId = ZoneId.of(timezone);
        ZonedDateTime now = ZonedDateTime.now(zoneId);
        
        // Check if user has preferred delivery time
        if (preferences.getPreferredDeliveryTime() != null) {
            ZonedDateTime preferredTime = calculateNextPreferredTime(
                preferences.getPreferredDeliveryTime(), 
                zoneId,
                priority
            );
            
            if (preferredTime != null) {
                return preferredTime;
            }
        }
        
        // Analyze user activity patterns
        ZonedDateTime optimalTime = analyzeActivityPatterns(userId, presence, zoneId);
        
        // For high priority, don't delay more than 1 hour
        if (priority == Priority.HIGH) {
            ZonedDateTime maxDelay = now.plusHours(1);
            if (optimalTime.isAfter(maxDelay)) {
                return maxDelay;
            }
        }
        
        // For medium priority, don't delay more than 4 hours
        if (priority == Priority.MEDIUM) {
            ZonedDateTime maxDelay = now.plusHours(4);
            if (optimalTime.isAfter(maxDelay)) {
                return maxDelay;
            }
        }
        
        return optimalTime;
    }
    
    private ZonedDateTime calculateNextPreferredTime(PreferredDeliveryTime preferredTime,
                                                    ZoneId zoneId,
                                                    Priority priority) {
        ZonedDateTime now = ZonedDateTime.now(zoneId);
        LocalTime startTime = LocalTime.parse(preferredTime.getStartTime());
        LocalTime endTime = LocalTime.parse(preferredTime.getEndTime());
        
        // Check if current time is within preferred window
        LocalTime currentTime = now.toLocalTime();
        if (currentTime.isAfter(startTime) && currentTime.isBefore(endTime)) {
            // We're in the preferred window, deliver now
            return now;
        }
        
        // Calculate next preferred window
        ZonedDateTime nextWindow = now.with(startTime);
        if (currentTime.isAfter(endTime)) {
            // Preferred window has passed today, schedule for tomorrow
            nextWindow = nextWindow.plusDays(1);
        }
        
        // Check if the day is preferred
        if (preferredTime.getPreferredDays() != null && !preferredTime.getPreferredDays().isEmpty()) {
            String dayOfWeek = nextWindow.getDayOfWeek().toString().substring(0, 3).toUpperCase();
            if (!preferredTime.getPreferredDays().contains(dayOfWeek)) {
                // Find next preferred day
                for (int i = 1; i <= 7; i++) {
                    nextWindow = nextWindow.plusDays(1);
                    dayOfWeek = nextWindow.getDayOfWeek().toString().substring(0, 3).toUpperCase();
                    if (preferredTime.getPreferredDays().contains(dayOfWeek)) {
                        break;
                    }
                }
            }
        }
        
        return nextWindow;
    }
    
    private ZonedDateTime analyzeActivityPatterns(String userId, 
                                                 UserPresence presence,
                                                 ZoneId zoneId) {
        ZonedDateTime now = ZonedDateTime.now(zoneId);
        
        // If user is currently active, deliver now
        if (presence != null && presence.getChannelPresence() != null) {
            boolean isActive = presence.getChannelPresence().values().stream()
                .anyMatch(status -> status.isActive() || status.isOnline());
            
            if (isActive) {
                return now;
            }
        }
        
        // Analyze historical activity patterns (simplified version)
        // In production, this would query historical data
        LocalTime[] commonActiveHours = {
            LocalTime.of(9, 0),   // Morning
            LocalTime.of(12, 30), // Lunch
            LocalTime.of(15, 0),  // Afternoon
            LocalTime.of(18, 0)   // Evening
        };
        
        for (LocalTime activeHour : commonActiveHours) {
            ZonedDateTime potentialTime = now.with(activeHour);
            if (potentialTime.isAfter(now)) {
                return potentialTime;
            }
        }
        
        // Default to next morning if no pattern found
        return now.plusDays(1).with(LocalTime.of(9, 0));
    }
    
    public void scheduleMessage(ChannelMessage message, ZonedDateTime scheduledTime) {
        ScheduledNotification scheduled = ScheduledNotification.builder()
            .messageId(message.getMessageId())
            .channelMessage(message)
            .scheduledTime(scheduledTime)
            .status(ScheduledStatus.PENDING)
            .createdAt(Instant.now())
            .build();
        
        mongoTemplate.save(scheduled, "scheduled_notifications");
        log.info("Scheduled notification {} for delivery at {}", 
            message.getMessageId(), scheduledTime);
    }
    
    @Scheduled(fixedDelay = 60000) // Check every minute
    public void processScheduledNotifications() {
        ZonedDateTime now = ZonedDateTime.now(ZoneOffset.UTC);
        
        Query query = new Query(Criteria.where("scheduledTime").lte(now)
            .and("status").is(ScheduledStatus.PENDING));
        
        List<ScheduledNotification> dueNotifications = 
            mongoTemplate.find(query, ScheduledNotification.class, "scheduled_notifications");
        
        for (ScheduledNotification notification : dueNotifications) {
            try {
                // Publish to EventBridge
                eventBridgePublisher.publish(notification.getChannelMessage());
                
                // Update status
                notification.setStatus(ScheduledStatus.SENT);
                notification.setProcessedAt(Instant.now());
                mongoTemplate.save(notification, "scheduled_notifications");
                
                log.info("Processed scheduled notification: {}", notification.getMessageId());
                
            } catch (Exception e) {
                log.error("Failed to process scheduled notification: {}", 
                    notification.getMessageId(), e);
                
                // Update retry count and status
                notification.setRetryCount(notification.getRetryCount() + 1);
                if (notification.getRetryCount() >= 3) {
                    notification.setStatus(ScheduledStatus.FAILED);
                }
                mongoTemplate.save(notification, "scheduled_notifications");
            }
        }
    }
}

@Data
@Builder
class ScheduledNotification {
    private String messageId;
    private ChannelMessage channelMessage;
    private ZonedDateTime scheduledTime;
    private ScheduledStatus status;
    private Instant createdAt;
    private Instant processedAt;
    private int retryCount;
}

enum ScheduledStatus {
    PENDING,
    SENT,
    FAILED,
    CANCELLED
}

// Enhanced Configuration
package com.notification.platform.routing.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.web.client.RestTemplateBuilder;
import org.springframework.cloud.circuitbreaker.resilience4j.Resilience4JCircuitBreakerFactory;
import org.springframework.cloud.circuitbreaker.resilience4j.Resilience4JConfigBuilder;
import org.springframework.cloud.client.circuitbreaker.Customizer;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.client.RestTemplate;
import software.amazon.awssdk.auth.credentials.DefaultCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.sqs.SqsClient;
import software.amazon.awssdk.services.eventbridge.EventBridgeClient;
import org.springframework.cache.CacheManager;
import org.springframework.cache.caffeine.CaffeineCacheManager;
import com.github.benmanes.caffeine.cache.Caffeine;
import io.github.resilience4j.circuitbreaker.CircuitBreakerConfig;
import io.github.resilience4j.timelimiter.TimeLimiterConfig;
import java.time.Duration;
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
    public RestTemplate restTemplate(RestTemplateBuilder builder) {
        return builder
            .setConnectTimeout(Duration.ofSeconds(5))
            .setReadTimeout(Duration.ofSeconds(10))
            .build();
    }
    
    @Bean
    public Customizer<Resilience4JCircuitBreakerFactory> defaultCustomizer() {
        return factory -> factory.configureDefault(id -> new Resilience4JConfigBuilder(id)
            .timeLimiterConfig(TimeLimiterConfig.custom()
                .timeoutDuration(Duration.ofSeconds(10))
                .build())
            .circuitBreakerConfig(CircuitBreakerConfig.custom()
                .slidingWindowSize(10)
                .minimumNumberOfCalls(5)
                .permittedNumberOfCallsInHalfOpenState(3)
                .automaticTransitionFromOpenToHalfOpenEnabled(true)
                .waitDurationInOpenState(Duration.ofSeconds(30))
                .failureRateThreshold(50)
                .build())
            .build());
    }
    
    @Bean
    public CacheManager cacheManager() {
        CaffeineCacheManager cacheManager = new CaffeineCacheManager();
        
        // Configure different caches with different settings
        cacheManager.registerCustomCache("employeeData",
            Caffeine.newBuilder()
                .maximumSize(10000)
                .expireAfterWrite(1, TimeUnit.HOURS)
                .build());
        
        cacheManager.registerCustomCache("userPresence",
            Caffeine.newBuilder()
                .maximumSize(10000)
                .expireAfterWrite(5, TimeUnit.MINUTES)
                .build());
        
        cacheManager.registerCustomCache("templates",
            Caffeine.newBuilder()
                .maximumSize(1000)
                .expireAfterWrite(30, TimeUnit.MINUTES)
                .build());
        
        // Default cache configuration
        cacheManager.setCaffeine(Caffeine.newBuilder()
            .maximumSize(5000)
            .expireAfterWrite(10, TimeUnit.MINUTES));
        
        return cacheManager;
    }
}

// SQS Message Listener remains the same as before
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
import java.util.Map;
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
            
            // Parse the message
            Map<String, Object> messageBody = objectMapper.readValue(message.body(), Map.class);
            
            // Check for smart notification flag
            boolean smartNotification = messageBody.containsKey("smartNotification") ? 
                (Boolean) messageBody.get("smartNotification") : false;
            
            // Build routing context
            RoutingContext context = RoutingContext.builder()
                .notificationId((String) messageBody.get("notificationId"))
                .userId((String) messageBody.get("userId"))
                .capability((String) messageBody.get("capability"))
                .payload((Map<String, Object>) messageBody.get("payload"))
                .metadata((Map<String, String>) messageBody.get("metadata"))
                .priority(Priority.valueOf((String) messageBody.get("priority")))
                .correlationId((String) messageBody.get("correlationId"))
                .smartNotification(smartNotification)
                .timestamp(Instant.now())
                .build();
            
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

// EventBridge Publisher remains the same
package com.notification.platform.routing.queue;

import com.notification.platform.routing.domain.ChannelMessage;
import com.notification.platform.routing.domain.ChannelType;
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

class EventBridgePublishException extends RuntimeException {
    public EventBridgePublishException(String message) {
        super(message);
    }
    
    public EventBridgePublishException(String message, Throwable cause) {
        super(message, cause);
    }
}

// Updated Application Properties
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

external:
  services:
    user-presence:
      url: ${USER_PRESENCE_SERVICE_URL:http://user-presence-service:8080}
    employee-data:
      url: ${EMPLOYEE_DATA_SERVICE_URL:http://employee-service:8080}
    bff:
      url: ${BFF_SERVICE_URL:http://bff-service:8080}

resilience4j:
  circuitbreaker:
    instances:
      user-presence:
        sliding-window-size: 10
        minimum-number-of-calls: 5
        failure-rate-threshold: 50
        wait-duration-in-open-state: 30s
      employee-data:
        sliding-window-size: 10
        minimum-number-of-calls: 5
        failure-rate-threshold: 50
        wait-duration-in-open-state: 30s
      bff-service:
        sliding-window-size: 10
        minimum-number-of-calls: 5
        failure-rate-threshold: 50
        wait-duration-in-open-state: 30s

management:
  endpoints:
    web:
      exposure:
        include: health,info,metrics,prometheus
  metrics:
    export:
      prometheus:
        enabled: true
  health:
    circuitbreakers:
      enabled: true

logging:
  level:
    com.notification.platform: DEBUG
    io.github.resilience4j: DEBUG
