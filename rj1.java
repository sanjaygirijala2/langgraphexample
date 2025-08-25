// ==================== APPLICATION MAIN CLASS ====================
package com.notification.platform.router;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cache.annotation.EnableCaching;
import org.springframework.scheduling.annotation.EnableAsync;

@SpringBootApplication
@EnableCaching
@EnableAsync
public class RouterServiceApplication {
    public static void main(String[] args) {
        SpringApplication.run(RouterServiceApplication.class, args);
    }
}

// ==================== DOMAIN MODELS ====================
package com.notification.platform.router.domain;

import lombok.Data;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

@Data
@Document(collection = "notifications")
public class NotificationInfo {
    @Id
    private String id;
    private String capabilityId;
    private List<String> recipients;
    private Map<String, Object> content;
    private Map<String, Object> metadata;
    private LocalDateTime createdAt;
    private String status;
    private Integer priority;
}

@Data
public class Capability {
    private String id;
    private String name;
    private List<Route> routes;
    private Map<String, Object> config;
    private boolean active;
}

@Data
public class Route {
    private String id;
    private String type; // MOBILE, WEB, EMAIL, TEAMS
    private String templateId;
    private Integer priority;
    private Map<String, Object> config;
}

@Data
public class RoutingContext {
    private String notificationId;
    private String capabilityName;
    private String recipientId;
    private List<String> routes;
    private Map<String, Object> personalizedContent;
    private Map<String, Object> metadata;
    private LocalDateTime timestamp;
    
    public static Builder builder() {
        return new Builder();
    }
    
    public static class Builder {
        private RoutingContext context = new RoutingContext();
        
        public Builder notificationId(String id) {
            context.notificationId = id;
            return this;
        }
        
        public Builder capabilityName(String name) {
            context.capabilityName = name;
            return this;
        }
        
        public Builder recipientId(String id) {
            context.recipientId = id;
            return this;
        }
        
        public Builder routes(List<String> routes) {
            context.routes = routes;
            return this;
        }
        
        public Builder personalizedContent(Map<String, Object> content) {
            context.personalizedContent = content;
            return this;
        }
        
        public Builder metadata(Map<String, Object> metadata) {
            context.metadata = metadata;
            return this;
        }
        
        public RoutingContext build() {
            context.timestamp = LocalDateTime.now();
            return context;
        }
    }
}

@Data
public class UserPreference {
    private String userId;
    private Map<String, Boolean> channelPreferences;
    private List<String> blockedCapabilities;
    private String timezone;
    private Map<String, Object> customSettings;
}

@Data
public class DeviceInfo {
    private String userId;
    private String deviceId;
    private String deviceType; // IOS, ANDROID, WEB
    private String token;
    private boolean active;
    private LocalDateTime lastActiveAt;
}

// ==================== FILTER FRAMEWORK ====================
package com.notification.platform.router.filter;

import com.notification.platform.router.domain.RoutingContext;

public interface RoutingFilter {
    boolean shouldProcess(RoutingContext context);
    RoutingContext process(RoutingContext context);
    int getOrder();
    String getName();
}

package com.notification.platform.router.filter;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import java.util.Comparator;
import java.util.List;

@Slf4j
@Component
public class RoutingFilterChain {
    private final List<RoutingFilter> filters;
    
    public RoutingFilterChain(List<RoutingFilter> filters) {
        this.filters = filters;
        this.filters.sort(Comparator.comparingInt(RoutingFilter::getOrder));
        log.info("Initialized filter chain with {} filters", filters.size());
    }
    
    public RoutingContext execute(RoutingContext context) {
        log.debug("Executing filter chain for notification: {}", context.getNotificationId());
        
        for (RoutingFilter filter : filters) {
            if (filter.shouldProcess(context)) {
                log.debug("Processing filter: {}", filter.getName());
                context = filter.process(context);
                
                if (context == null) {
                    log.warn("Filter {} returned null context, stopping chain", filter.getName());
                    break;
                }
            } else {
                log.debug("Skipping filter: {}", filter.getName());
            }
        }
        
        return context;
    }
}

// ==================== FILTER IMPLEMENTATIONS ====================
package com.notification.platform.router.filter.impl;

import com.notification.platform.router.domain.RoutingContext;
import com.notification.platform.router.domain.UserPreference;
import com.notification.platform.router.filter.RoutingFilter;
import com.notification.platform.router.service.UserPreferenceService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import java.util.ArrayList;
import java.util.List;

@Slf4j
@Component
@RequiredArgsConstructor
public class UserPreferenceFilter implements RoutingFilter {
    
    private final UserPreferenceService preferenceService;
    
    @Override
    public boolean shouldProcess(RoutingContext context) {
        return context != null && context.getRecipientId() != null;
    }
    
    @Override
    public RoutingContext process(RoutingContext context) {
        try {
            UserPreference preference = preferenceService.getUserPreference(context.getRecipientId());
            
            if (preference == null) {
                log.debug("No user preference found for: {}", context.getRecipientId());
                return context;
            }
            
            // Check if capability is blocked
            if (preference.getBlockedCapabilities() != null &&
                preference.getBlockedCapabilities().contains(context.getCapabilityName())) {
                log.info("Capability {} blocked for user {}", 
                    context.getCapabilityName(), context.getRecipientId());
                return null;
            }
            
            // Filter routes based on channel preferences
            if (preference.getChannelPreferences() != null && context.getRoutes() != null) {
                List<String> filteredRoutes = new ArrayList<>();
                
                for (String route : context.getRoutes()) {
                    String channelType = extractChannelType(route);
                    if (preference.getChannelPreferences().getOrDefault(channelType, true)) {
                        filteredRoutes.add(route);
                    } else {
                        log.debug("Channel {} disabled for user {}", channelType, context.getRecipientId());
                    }
                }
                
                context.setRoutes(filteredRoutes);
            }
            
            // Add preference metadata
            context.getMetadata().put("userTimezone", preference.getTimezone());
            context.getMetadata().put("userSettings", preference.getCustomSettings());
            
        } catch (Exception e) {
            log.error("Error processing user preference filter", e);
            // Continue with original context on error
        }
        
        return context;
    }
    
    private String extractChannelType(String route) {
        // Extract channel type from route (e.g., "MOBILE_IOS" -> "MOBILE")
        if (route.contains("_")) {
            return route.substring(0, route.indexOf("_"));
        }
        return route;
    }
    
    @Override
    public int getOrder() {
        return 10;
    }
    
    @Override
    public String getName() {
        return "UserPreferenceFilter";
    }
}

package com.notification.platform.router.filter.impl;

import com.notification.platform.router.domain.DeviceInfo;
import com.notification.platform.router.domain.RoutingContext;
import com.notification.platform.router.filter.RoutingFilter;
import com.notification.platform.router.service.DeviceService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Slf4j
@Component
@RequiredArgsConstructor
public class DeviceMappingFilter implements RoutingFilter {
    
    private final DeviceService deviceService;
    private static final int INACTIVE_DEVICE_DAYS = 30;
    
    @Override
    public boolean shouldProcess(RoutingContext context) {
        return context != null && 
               context.getRoutes() != null && 
               context.getRoutes().stream().anyMatch(r -> r.startsWith("MOBILE"));
    }
    
    @Override
    public RoutingContext process(RoutingContext context) {
        try {
            List<DeviceInfo> devices = deviceService.getActiveDevices(context.getRecipientId());
            
            if (devices.isEmpty()) {
                log.warn("No active devices found for user: {}", context.getRecipientId());
                // Remove mobile routes if no devices
                context.setRoutes(filterOutMobileRoutes(context.getRoutes()));
                return context;
            }
            
            List<String> updatedRoutes = new ArrayList<>();
            Map<String, List<DeviceInfo>> devicesByType = groupDevicesByType(devices);
            
            for (String route : context.getRoutes()) {
                if (route.startsWith("MOBILE")) {
                    // Add specific device routes
                    if (devicesByType.containsKey("IOS") && 
                        (route.equals("MOBILE") || route.equals("MOBILE_IOS"))) {
                        updatedRoutes.add("MOBILE_IOS");
                    }
                    if (devicesByType.containsKey("ANDROID") && 
                        (route.equals("MOBILE") || route.equals("MOBILE_ANDROID"))) {
                        updatedRoutes.add("MOBILE_ANDROID");
                    }
                } else {
                    updatedRoutes.add(route);
                }
            }
            
            context.setRoutes(updatedRoutes);
            
            // Add device tokens to metadata
            List<String> deviceTokens = devices.stream()
                .filter(d -> d.isActive() && isRecentlyActive(d))
                .map(DeviceInfo::getToken)
                .toList();
            
            context.getMetadata().put("deviceTokens", deviceTokens);
            context.getMetadata().put("deviceCount", devices.size());
            
        } catch (Exception e) {
            log.error("Error processing device mapping filter", e);
        }
        
        return context;
    }
    
    private List<String> filterOutMobileRoutes(List<String> routes) {
        return routes.stream()
            .filter(r -> !r.startsWith("MOBILE"))
            .toList();
    }
    
    private Map<String, List<DeviceInfo>> groupDevicesByType(List<DeviceInfo> devices) {
        return devices.stream()
            .filter(DeviceInfo::isActive)
            .filter(this::isRecentlyActive)
            .collect(java.util.stream.Collectors.groupingBy(DeviceInfo::getDeviceType));
    }
    
    private boolean isRecentlyActive(DeviceInfo device) {
        return device.getLastActiveAt() != null &&
               device.getLastActiveAt().isAfter(LocalDateTime.now().minusDays(INACTIVE_DEVICE_DAYS));
    }
    
    @Override
    public int getOrder() {
        return 20;
    }
    
    @Override
    public String getName() {
        return "DeviceMappingFilter";
    }
}

package com.notification.platform.router.filter.impl;

import com.notification.platform.router.domain.RoutingContext;
import com.notification.platform.router.filter.RoutingFilter;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import java.time.LocalTime;
import java.time.ZoneId;
import java.time.ZonedDateTime;

@Slf4j
@Component
public class QuietHoursFilter implements RoutingFilter {
    
    @Value("${notification.quiet-hours.enabled:true}")
    private boolean enabled;
    
    @Value("${notification.quiet-hours.start:22}")
    private int quietHourStart;
    
    @Value("${notification.quiet-hours.end:8}")
    private int quietHourEnd;
    
    @Override
    public boolean shouldProcess(RoutingContext context) {
        return enabled && context != null;
    }
    
    @Override
    public RoutingContext process(RoutingContext context) {
        String timezone = (String) context.getMetadata().get("userTimezone");
        if (timezone == null) {
            timezone = "UTC";
        }
        
        ZonedDateTime userTime = ZonedDateTime.now(ZoneId.of(timezone));
        LocalTime currentTime = userTime.toLocalTime();
        
        boolean inQuietHours = isInQuietHours(currentTime);
        
        if (inQuietHours) {
            Integer priority = (Integer) context.getMetadata().get("priority");
            if (priority == null || priority < 3) { // Only high priority (3+) bypasses quiet hours
                log.info("Notification {} delayed due to quiet hours for user {}", 
                    context.getNotificationId(), context.getRecipientId());
                
                // Add delay metadata
                context.getMetadata().put("delayedUntil", calculateDelayUntil(userTime));
                context.getMetadata().put("delayReason", "QUIET_HOURS");
                
                // You might want to send to a delayed queue instead
                return null;
            }
        }
        
        return context;
    }
    
    private boolean isInQuietHours(LocalTime time) {
        int hour = time.getHour();
        if (quietHourStart > quietHourEnd) {
            // Quiet hours span midnight
            return hour >= quietHourStart || hour < quietHourEnd;
        } else {
            return hour >= quietHourStart && hour < quietHourEnd;
        }
    }
    
    private String calculateDelayUntil(ZonedDateTime userTime) {
        ZonedDateTime delayUntil = userTime.withHour(quietHourEnd).withMinute(0).withSecond(0);
        if (delayUntil.isBefore(userTime)) {
            delayUntil = delayUntil.plusDays(1);
        }
        return delayUntil.toString();
    }
    
    @Override
    public int getOrder() {
        return 30;
    }
    
    @Override
    public String getName() {
        return "QuietHoursFilter";
    }
}

// ==================== MAIN ROUTING SERVICE ====================
package com.notification.platform.router.service;

import com.notification.platform.router.domain.*;
import com.notification.platform.router.filter.RoutingFilterChain;
import io.awspring.cloud.sqs.annotation.SqsListener;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
public class RouterService {
    
    private final NotificationRepository notificationRepository;
    private final CapabilityService capabilityService;
    private final RoutingFilterChain filterChain;
    private final EventBridgePublisher eventBridgePublisher;
    private final TemplateService templateService;
    private final BatchProcessor batchProcessor;
    
    @SqsListener("${aws.sqs.router-queue}")
    public void processNotificationMessage(NotificationMessage message) {
        log.info("Processing notification message with id: {}", message.getNotificationId());
        
        try {
            // Validate and check expiry
            if (isNotificationExpired(message)) {
                log.warn("Notification {} has expired", message.getNotificationId());
                return;
            }
            
            // Fetch notification from MongoDB
            NotificationInfo notification = notificationRepository.findById(message.getNotificationId())
                .orElseThrow(() -> new RuntimeException("Notification not found: " + message.getNotificationId()));
            
            // Get capability and routes
            Capability capability = capabilityService.getCapability(notification.getCapabilityId());
            if (capability == null || !capability.isActive()) {
                log.warn("Capability {} not found or inactive", notification.getCapabilityId());
                return;
            }
            
            // Determine routes
            Set<String> routes = determineRoutes(capability);
            if (routes.isEmpty()) {
                log.warn("No routes found for capability: {}", capability.getName());
                return;
            }
            
            // Process recipients in batches
            List<String> recipients = notification.getRecipients();
            if (recipients == null || recipients.isEmpty()) {
                log.warn("No recipients found for notification: {}", notification.getId());
                return;
            }
            
            log.info("Processing {} recipients for notification {} with routes: {}", 
                recipients.size(), notification.getId(), routes);
            
            // Process in batches for large recipient lists
            batchProcessor.processBatches(recipients, 100, batch -> 
                processRecipientBatch(batch, notification, capability, new ArrayList<>(routes))
            );
            
        } catch (Exception e) {
            log.error("Error processing notification message: {}", message.getNotificationId(), e);
            throw new RuntimeException("Failed to process notification", e);
        }
    }
    
    private void processRecipientBatch(List<String> recipients, NotificationInfo notification, 
                                      Capability capability, List<String> routes) {
        List<CompletableFuture<Void>> futures = new ArrayList<>();
        
        for (String recipientId : recipients) {
            CompletableFuture<Void> future = CompletableFuture.runAsync(() -> {
                try {
                    processRecipient(recipientId, notification, capability, routes);
                } catch (Exception e) {
                    log.error("Error processing recipient: {}", recipientId, e);
                }
            });
            futures.add(future);
        }
        
        // Wait for all recipients in batch to complete
        CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).join();
    }
    
    private void processRecipient(String recipientId, NotificationInfo notification, 
                                 Capability capability, List<String> routes) {
        log.debug("Processing recipient: {}", recipientId);
        
        // Build routing context
        RoutingContext context = RoutingContext.builder()
            .notificationId(notification.getId())
            .capabilityName(capability.getName())
            .recipientId(recipientId)
            .routes(routes)
            .personalizedContent(new HashMap<>(notification.getContent()))
            .metadata(new HashMap<>(notification.getMetadata()))
            .build();
        
        // Add priority from notification
        context.getMetadata().put("priority", notification.getPriority());
        
        // Execute filter chain
        context = filterChain.execute(context);
        
        if (context == null || context.getRoutes() == null || context.getRoutes().isEmpty()) {
            log.debug("No routes after filtering for recipient: {}", recipientId);
            return;
        }
        
        // Process each route
        for (String route : context.getRoutes()) {
            try {
                processRoute(route, context, capability);
            } catch (Exception e) {
                log.error("Error processing route {} for recipient {}", route, recipientId, e);
            }
        }
    }
    
    private void processRoute(String route, RoutingContext context, Capability capability) {
        // Get template for route
        String templateId = getTemplateIdForRoute(route, capability);
        
        // Personalize content with template
        Map<String, Object> personalizedContent = templateService.personalizeContent(
            templateId, 
            context.getRecipientId(), 
            context.getPersonalizedContent()
        );
        
        // Build event for EventBridge
        RoutingEvent event = RoutingEvent.builder()
            .notificationId(context.getNotificationId())
            .recipientId(context.getRecipientId())
            .route(route)
            .content(personalizedContent)
            .metadata(context.getMetadata())
            .timestamp(context.getTimestamp())
            .build();
        
        // Publish to EventBridge
        eventBridgePublisher.publish(event, route);
        
        log.debug("Published event for route {} to recipient {}", route, context.getRecipientId());
    }
    
    private boolean isNotificationExpired(NotificationMessage message) {
        // Check if notification has expired based on timestamp
        if (message.getExpiryTime() != null) {
            return message.getExpiryTime().isBefore(LocalDateTime.now());
        }
        return false;
    }
    
    private Set<String> determineRoutes(Capability capability) {
        return capability.getRoutes().stream()
            .filter(r -> r.getConfig() != null && 
                        Boolean.TRUE.equals(r.getConfig().get("enabled")))
            .map(Route::getType)
            .collect(Collectors.toSet());
    }
    
    private String getTemplateIdForRoute(String route, Capability capability) {
        return capability.getRoutes().stream()
            .filter(r -> r.getType().equals(route))
            .map(Route::getTemplateId)
            .findFirst()
            .orElse(null);
    }
}

// ==================== BATCH PROCESSOR ====================
package com.notification.platform.router.service;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

@Slf4j
@Component
public class BatchProcessor {
    
    public <T> void processBatches(List<T> items, int batchSize, Consumer<List<T>> processor) {
        if (items == null || items.isEmpty()) {
            return;
        }
        
        int totalBatches = (items.size() + batchSize - 1) / batchSize;
        log.info("Processing {} items in {} batches of size {}", items.size(), totalBatches, batchSize);
        
        for (int i = 0; i < items.size(); i += batchSize) {
            int end = Math.min(i + batchSize, items.size());
            List<T> batch = new ArrayList<>(items.subList(i, end));
            
            try {
                log.debug("Processing batch {}/{} with {} items", 
                    (i / batchSize) + 1, totalBatches, batch.size());
                processor.accept(batch);
            } catch (Exception e) {
                log.error("Error processing batch", e);
            }
        }
    }
}

// ==================== EVENTBRIDGE PUBLISHER ====================
package com.notification.platform.router.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.notification.platform.router.domain.RoutingEvent;
import io.github.resilience4j.circuitbreaker.annotation.CircuitBreaker;
import io.github.resilience4j.retry.annotation.Retry;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import software.amazon.awssdk.services.eventbridge.EventBridgeClient;
import software.amazon.awssdk.services.eventbridge.model.PutEventsRequest;
import software.amazon.awssdk.services.eventbridge.model.PutEventsRequestEntry;
import software.amazon.awssdk.services.eventbridge.model.PutEventsResponse;

@Slf4j
@Service
@RequiredArgsConstructor
public class EventBridgePublisher {
    
    private final EventBridgeClient eventBridgeClient;
    private final ObjectMapper objectMapper;
    
    @Value("${aws.eventbridge.bus-name}")
    private String eventBusName;
    
    @Value("${aws.eventbridge.source}")
    private String eventSource;
    
    @CircuitBreaker(name = "eventbridge", fallbackMethod = "publishFallback")
    @Retry(name = "eventbridge")
    public void publish(RoutingEvent event, String route) {
        try {
            String eventJson = objectMapper.writeValueAsString(event);
            
            PutEventsRequestEntry entry = PutEventsRequestEntry.builder()
                .source(eventSource)
                .detailType(getDetailType(route))
                .detail(eventJson)
                .eventBusName(eventBusName)
                .build();
            
            PutEventsRequest request = PutEventsRequest.builder()
                .entries(entry)
                .build();
            
            PutEventsResponse response = eventBridgeClient.putEvents(request);
            
            if (response.failedEntryCount() > 0) {
                log.error("Failed to publish {} events to EventBridge", response.failedEntryCount());
                throw new RuntimeException("EventBridge publish failed");
            }
            
            log.debug("Successfully published event to EventBridge for route: {}", route);
            
        } catch (Exception e) {
            log.error("Error publishing to EventBridge", e);
            throw new RuntimeException("Failed to publish event", e);
        }
    }
    
    public void publishFallback(RoutingEvent event, String route, Exception ex) {
        log.error("EventBridge publish failed, using fallback for route: {}", route, ex);
        // Could implement DLQ or alternative processing here
    }
    
    private String getDetailType(String route) {
        return "notification." + route.toLowerCase();
    }
}

// ==================== SUPPORTING SERVICES ====================
package com.notification.platform.router.service;

import com.notification.platform.router.domain.Capability;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.data.mongodb.core.query.Criteria;
import org.springframework.data.mongodb.core.query.Query;
import org.springframework.stereotype.Service;

@Slf4j
@Service
@RequiredArgsConstructor
public class CapabilityService {
    
    private final MongoTemplate mongoTemplate;
    
    @Cacheable(value = "capabilities", key = "#capabilityId")
    public Capability getCapability(String capabilityId) {
        Query query = new Query(Criteria.where("id").is(capabilityId));
        return mongoTemplate.findOne(query, Capability.class, "capabilities");
    }
}

package com.notification.platform.router.service;

import com.notification.platform.router.domain.UserPreference;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.data.mongodb.core.query.Criteria;
import org.springframework.data.mongodb.core.query.Query;
import org.springframework.stereotype.Service;

@Slf4j
@Service
@RequiredArgsConstructor
public class UserPreferenceService {
    
    private final MongoTemplate mongoTemplate;
    
    @Cacheable(value = "userPreferences", key = "#userId")
    public UserPreference getUserPreference(String userId) {
        Query query = new Query(Criteria.where("userId").is(userId));
        return mongoTemplate.findOne(query, UserPreference.class, "user_preferences");
    }
}

package com.notification.platform.router.service;

import com.notification.platform.router.domain.DeviceInfo;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.data.mongodb.core.query.Criteria;
import org.springframework.data.mongodb.core.query.Query;
import org.springframework.stereotype.Service;
import java.util.List;

@Slf4j
@Service
@RequiredArgsConstructor
public class DeviceService {
    
    private final MongoTemplate mongoTemplate;
    
    @Cacheable(value = "devices", key = "#userId")
    public List<DeviceInfo> getActiveDevices(String userId) {
        Query query = new Query(Criteria.where("userId").is(userId)
            .and("active").is(true));
        return mongoTemplate.find(query, DeviceInfo.class, "devices");
    }
}

package com.notification.platform.router.service;

import freemarker.template.Configuration;
import freemarker.template.Template;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.data.mongodb.core.query.Criteria;
import org.springframework.data.mongodb.core.query.Query;
import org.springframework.stereotype.Service;
import java.io.StringWriter;
import java.util.HashMap;
import java.util.Map;

@Slf4j
@Service
@RequiredArgsConstructor
public class TemplateService {
    
    private final MongoTemplate mongoTemplate;
    private final Configuration freemarkerConfig;
    
    public Map<String, Object> personalizeContent(String templateId, String recipientId, 
                                                  Map<String, Object> content) {
        try {
            // Get template from DB
            NotificationTemplate template = getTemplate(templateId);
            
            if (template == null || !template.isDynamic()) {
                return content;
            }
            
            // Get user data for personalization
            Map<String, Object> userData = getUserData(recipientId);
            
            // Merge content with user data
            Map<String, Object> model = new HashMap<>();
            model.put("user", userData);
            model.put("content", content);
            
            // Process template
            Map<String, Object> personalizedContent = new HashMap<>(content);
            
            if (template.getSubjectTemplate() != null) {
                personalizedContent.put("subject", processTemplate(template.getSubjectTemplate(), model));
            }
            
            if (template.getBodyTemplate() != null) {
                personalizedContent.put("body", processTemplate(template.getBodyTemplate(), model));
            }
            
            return personalizedContent;
            
        } catch (Exception e) {
            log.error("Error personalizing content with template: {}", templateId, e);
            return content;
        }
    }
    
    @Cacheable(value = "templates", key = "#templateId")
    private NotificationTemplate getTemplate(String templateId) {
        Query query = new Query(Criteria.where("id").is(templateId));
        return mongoTemplate.findOne(query, NotificationTemplate.class, "templates");
    }
    
    private Map<String, Object> getUserData(String userId) {
        Query query = new Query(Criteria.where("id").is(userId));
        return mongoTemplate.findOne(query, Map.class, "users");
    }
    
    private String processTemplate(String templateStr, Map<String, Object> model) {
        try {
            Template template = new Template("template", templateStr, freemarkerConfig);
            StringWriter writer = new StringWriter();
            template.process(model, writer);
            return writer.toString();
        } catch (Exception e) {
            log.error("Error processing template", e);
            return templateStr;
        }
    }
}

// ==================== REPOSITORIES ====================
package com.notification.platform.router.repository;

import com.notification.platform.router.domain.NotificationInfo;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface NotificationRepository extends MongoRepository<NotificationInfo, String> {
}

// ==================== CONFIGURATION ====================
package com.notification.platform.router.config;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import freemarker.template.Configuration;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Primary;
import software.amazon.awssdk.auth.credentials.InstanceProfileCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.eventbridge.EventBridgeClient;
import software.amazon.awssdk.services.sqs.SqsAsyncClient;

@org.springframework.context.annotation.Configuration
public class AwsConfig {
    
    @Bean
    public EventBridgeClient eventBridgeClient() {
        return EventBridgeClient.builder()
            .region(Region.US_EAST_1)
            .credentialsProvider(InstanceProfileCredentialsProvider.create())
            .build();
    }
    
    @Bean
    public SqsAsyncClient sqsAsyncClient() {
        return SqsAsyncClient.builder()
            .region(Region.US_EAST_1)
            .credentialsProvider(InstanceProfileCredentialsProvider.create())
            .build();
    }
    
    @Bean
    @Primary
    public ObjectMapper objectMapper() {
        ObjectMapper mapper = new ObjectMapper();
        mapper.registerModule(new JavaTimeModule());
        return mapper;
    }
    
    @Bean
    public Configuration freemarkerConfiguration() {
        Configuration cfg = new Configuration(Configuration.VERSION_2_3_31);
        cfg.setDefaultEncoding("UTF-8");
        return cfg;
    }
}

package com.notification.platform.router.config;

import com.github.benmanes.caffeine.cache.Caffeine;
import org.springframework.cache.CacheManager;
import org.springframework.cache.caffeine.CaffeineCacheManager;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import java.util.concurrent.TimeUnit;

@Configuration
public class CacheConfig {
    
    @Bean
    public CacheManager cacheManager() {
        CaffeineCacheManager cacheManager = new CaffeineCacheManager(
            "capabilities", "userPreferences", "devices", "templates"
        );
        cacheManager.setCaffeine(Caffeine.newBuilder()
            .maximumSize(10000)
            .expireAfterWrite(5, TimeUnit.MINUTES));
        return cacheManager;
    }
}

// ==================== DTOs ====================
package com.notification.platform.router.domain;

import lombok.Builder;
import lombok.Data;
import java.time.LocalDateTime;
import java.util.Map;

@Data
@Builder
public class RoutingEvent {
    private String notificationId;
    private String recipientId;
    private String route;
    private Map<String, Object> content;
    private Map<String, Object> metadata;
    private LocalDateTime timestamp;
}

@Data
public class NotificationMessage {
    private String notificationId;
    private LocalDateTime expiryTime;
}

@Data
public class NotificationTemplate {
    private String id;
    private String name;
    private boolean dynamic;
    private String subjectTemplate;
    private String bodyTemplate;
    private Map<String, Object> config;













}
