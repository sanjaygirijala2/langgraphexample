// ==================== ROUTER SERVICE TEST ====================
package com.notification.platform.router.service;

import com.notification.platform.router.domain.*;
import com.notification.platform.router.filter.RoutingFilterChain;
import com.notification.platform.router.repository.NotificationRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import java.time.LocalDateTime;
import java.util.*;

import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class RouterServiceTest {
    
    @Mock
    private NotificationRepository notificationRepository;
    
    @Mock
    private CapabilityService capabilityService;
    
    @Mock
    private RoutingFilterChain filterChain;
    
    @Mock
    private EventBridgePublisher eventBridgePublisher;
    
    @Mock
    private TemplateService templateService;
    
    @Mock
    private BatchProcessor batchProcessor;
    
    @InjectMocks
    private RouterService routerService;
    
    private NotificationMessage notificationMessage;
    private NotificationInfo notificationInfo;
    private Capability capability;
    
    @BeforeEach
    void setUp() {
        notificationMessage = new NotificationMessage();
        notificationMessage.setNotificationId("test-notification-123");
        notificationMessage.setExpiryTime(LocalDateTime.now().plusHours(1));
        
        notificationInfo = new NotificationInfo();
        notificationInfo.setId("test-notification-123");
        notificationInfo.setCapabilityId("capability-1");
        notificationInfo.setRecipients(Arrays.asList("user1", "user2", "user3"));
        notificationInfo.setContent(Map.of("subject", "Test", "body", "Test Message"));
        notificationInfo.setMetadata(new HashMap<>());
        notificationInfo.setPriority(2);
        
        capability = new Capability();
        capability.setId("capability-1");
        capability.setName("TRANSACTIONAL");
        capability.setActive(true);
        
        Route mobileRoute = new Route();
        mobileRoute.setType("MOBILE");
        mobileRoute.setTemplateId("template-1");
        mobileRoute.setConfig(Map.of("enabled", true));
        
        Route emailRoute = new Route();
        emailRoute.setType("EMAIL");
        emailRoute.setTemplateId("template-2");
        emailRoute.setConfig(Map.of("enabled", true));
        
        capability.setRoutes(Arrays.asList(mobileRoute, emailRoute));
    }
    
    @Test
    void testProcessNotificationMessage_Success() {
        // Given
        when(notificationRepository.findById(anyString()))
            .thenReturn(Optional.of(notificationInfo));
        when(capabilityService.getCapability(anyString()))
            .thenReturn(capability);
        when(filterChain.execute(any(RoutingContext.class)))
            .thenAnswer(invocation -> invocation.getArgument(0));
        when(templateService.personalizeContent(anyString(), anyString(), any()))
            .thenReturn(Map.of("subject", "Personalized", "body", "Personalized Body"));
        
        doAnswer(invocation -> {
            List<String> batch = invocation.getArgument(0);
            Consumer<List<String>> processor = invocation.getArgument(2);
            processor.accept(batch);
            return null;
        }).when(batchProcessor).processBatches(any(), anyInt(), any());
        
        // When
        routerService.processNotificationMessage(notificationMessage);
        
        // Then
        verify(notificationRepository).findById("test-notification-123");
        verify(capabilityService).getCapability("capability-1");
        verify(filterChain, atLeastOnce()).execute(any(RoutingContext.class));
        verify(eventBridgePublisher, atLeastOnce()).publish(any(RoutingEvent.class), anyString());
    }
    
    @Test
    void testProcessNotificationMessage_ExpiredNotification() {
        // Given
        notificationMessage.setExpiryTime(LocalDateTime.now().minusHours(1));
        
        // When
        routerService.processNotificationMessage(notificationMessage);
        
        // Then
        verify(notificationRepository, never()).findById(anyString());
        verify(eventBridgePublisher, never()).publish(any(), anyString());
    }
    
    @Test
    void testProcessNotificationMessage_NotificationNotFound() {
        // Given
        when(notificationRepository.findById(anyString()))
            .thenReturn(Optional.empty());
        
        // When & Then
        assertThrows(RuntimeException.class, () -> 
            routerService.processNotificationMessage(notificationMessage));
    }
    
    @Test
    void testProcessNotificationMessage_InactiveCapability() {
        // Given
        capability.setActive(false);
        when(notificationRepository.findById(anyString()))
            .thenReturn(Optional.of(notificationInfo));
        when(capabilityService.getCapability(anyString()))
            .thenReturn(capability);
        
        // When
        routerService.processNotificationMessage(notificationMessage);
        
        // Then
        verify(filterChain, never()).execute(any());
        verify(eventBridgePublisher, never()).publish(any(), anyString());
    }
}

// ==================== FILTER CHAIN TEST ====================
package com.notification.platform.router.filter;

import com.notification.platform.router.domain.RoutingContext;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class RoutingFilterChainTest {
    
    private RoutingFilterChain filterChain;
    private RoutingFilter filter1;
    private RoutingFilter filter2;
    private RoutingFilter filter3;
    
    @BeforeEach
    void setUp() {
        filter1 = mock(RoutingFilter.class);
        when(filter1.getOrder()).thenReturn(10);
        when(filter1.getName()).thenReturn("Filter1");
        when(filter1.shouldProcess(any())).thenReturn(true);
        when(filter1.process(any())).thenAnswer(invocation -> invocation.getArgument(0));
        
        filter2 = mock(RoutingFilter.class);
        when(filter2.getOrder()).thenReturn(5);
        when(filter2.getName()).thenReturn("Filter2");
        when(filter2.shouldProcess(any())).thenReturn(true);
        when(filter2.process(any())).thenAnswer(invocation -> invocation.getArgument(0));
        
        filter3 = mock(RoutingFilter.class);
        when(filter3.getOrder()).thenReturn(20);
        when(filter3.getName()).thenReturn("Filter3");
        when(filter3.shouldProcess(any())).thenReturn(false);
        
        List<RoutingFilter> filters = Arrays.asList(filter1, filter2, filter3);
        filterChain = new RoutingFilterChain(filters);
    }
    
    @Test
    void testFilterChainExecutionOrder() {
        // Given
        RoutingContext context = RoutingContext.builder()
            .notificationId("test-123")
            .build();
        
        // When
        RoutingContext result = filterChain.execute(context);
        
        // Then
        assertNotNull(result);
        
        // Verify filter2 (order 5) executed before filter1 (order 10)
        InOrder inOrder = inOrder(filter2, filter1);
        inOrder.verify(filter2).process(any());
        inOrder.verify(filter1).process(any());
        
        // Verify filter3 was not processed (shouldProcess returned false)
        verify(filter3, never()).process(any());
    }
    
    @Test
    void testFilterChainStopsOnNull() {
        // Given
        when(filter2.process(any())).thenReturn(null);
        
        RoutingContext context = RoutingContext.builder()
            .notificationId("test-123")
            .build();
        
        // When
        RoutingContext result = filterChain.execute(context);
        
        // Then
        assertNull(result);
        verify(filter2).process(any());
        verify(filter1, never()).process(any()); // Should not process after null
    }
}

// ==================== FILTER TESTS ====================
package com.notification.platform.router.filter.impl;

import com.notification.platform.router.domain.*;
import com.notification.platform.router.service.UserPreferenceService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class UserPreferenceFilterTest {
    
    @Mock
    private UserPreferenceService preferenceService;
    
    @InjectMocks
    private UserPreferenceFilter filter;
    
    private RoutingContext context;
    private UserPreference userPreference;
    
    @BeforeEach
    void setUp() {
        context = RoutingContext.builder()
            .notificationId("test-123")
            .recipientId("user-456")
            .capabilityName("MARKETING")
            .routes(new ArrayList<>(Arrays.asList("EMAIL", "MOBILE", "SMS")))
            .metadata(new HashMap<>())
            .build();
        
        userPreference = new UserPreference();
        userPreference.setUserId("user-456");
        userPreference.setChannelPreferences(Map.of(
            "EMAIL", true,
            "MOBILE", false,
            "SMS", true
        ));
        userPreference.setTimezone("America/New_York");
    }
    
    @Test
    void testFilterRemovesDisabledChannels() {
        // Given
        when(preferenceService.getUserPreference("user-456"))
            .thenReturn(userPreference);
        
        // When
        RoutingContext result = filter.process(context);
        
        // Then
        assertNotNull(result);
        assertEquals(2, result.getRoutes().size());
        assertTrue(result.getRoutes().contains("EMAIL"));
        assertTrue(result.getRoutes().contains("SMS"));
        assertFalse(result.getRoutes().contains("MOBILE"));
    }
    
    @Test
    void testFilterBlocksCapability() {
        // Given
        userPreference.setBlockedCapabilities(Arrays.asList("MARKETING", "PROMOTIONAL"));
        when(preferenceService.getUserPreference("user-456"))
            .thenReturn(userPreference);
        
        // When
        RoutingContext result = filter.process(context);
        
        // Then
        assertNull(result);
    }
    
    @Test
    void testFilterAddsUserMetadata() {
        // Given
        when(preferenceService.getUserPreference("user-456"))
            .thenReturn(userPreference);
        
        // When
        RoutingContext result = filter.process(context);
        
        // Then
        assertNotNull(result);
        assertEquals("America/New_York", result.getMetadata().get("userTimezone"));
    }
}

// ==================== INTEGRATION TEST ====================
package com.notification.platform.router;

import com.notification.platform.router.domain.NotificationInfo;
import com.notification.platform.router.repository.NotificationRepository;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.test.context.ActiveProfiles;
import org.testcontainers.containers.MongoDBContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import software.amazon.awssdk.services.eventbridge.EventBridgeClient;
import software.amazon.awssdk.services.sqs.SqsAsyncClient;

import java.util.Arrays;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest
@ActiveProfiles("test")
@Testcontainers
class RouterServiceIntegrationTest {
    
    @Container
    static MongoDBContainer mongoDBContainer = new MongoDBContainer("mongo:5.0")
        .withExposedPorts(27017);
    
    @Autowired
    private NotificationRepository notificationRepository;
    
    @MockBean
    private EventBridgeClient eventBridgeClient;
    
    @MockBean
    private SqsAsyncClient sqsAsyncClient;
    
    @Test
    void testCompleteRoutingFlow() {
        // Given
        NotificationInfo notification = new NotificationInfo();
        notification.setId("integration-test-123");
        notification.setCapabilityId("cap-1");
        notification.setRecipients(Arrays.asList("user1", "user2"));
        notification.setContent(Map.of("message", "Integration Test"));
        notification.setPriority(3);
        
        // When
        NotificationInfo saved = notificationRepository.save(notification);
        
        // Then
        assertNotNull(saved);
        assertEquals("integration-test-123", saved.getId());
        
        // Verify retrieval
        assertTrue(notificationRepository.findById("integration-test-123").isPresent());
    }
}
