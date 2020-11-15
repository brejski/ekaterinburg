// #include <vulkan/vulkan.h>
// This replacement is needed for showing something
// simple include is enough for off-screen rendering
#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN

#include <GLFW/glfw3.h>

// For reporting and propagating errors
#include <iostream>
#include <stdexcept>
// Provides EXIT_SUCCESS and EXIT_FAILURE macros
#include <cstdlib>

#include <optional>
#include <set>
#include <unordered_set>
#include <vector>
// #include <stdio.h>
#include <cstdint>
#include <algorithm>
#include <fstream>

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;
const int MAX_FRAMES_IN_FLIGHT = 2;

// swapchain is needed to synchronize the presentation of rendered
// images with the refresh rate of the screen
const std::vector<const char *> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME};

// all standard validation is bundled into this layer
const std::vector<const char *> validationLayers = {
    "VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG // standard C++ macro for not-debug mode
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

bool checkValidationLayerSupport() {
  uint32_t layerCount;
  vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

  std::vector<VkLayerProperties> availableLayers(layerCount);
  vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

  for (const char *layerName : validationLayers) {
    bool layerFound = false;

    for (const auto &layerProperties : availableLayers) {
      if (strcmp(layerName, layerProperties.layerName) == 0) {
        layerFound = true;
        break;
      }
    }
    if (!layerFound) {
      return false;
    }
  }
  return true;
}

std::vector<const char *> getRequiredExtensions() {
  // glfw returns the extensions it needs to do, that can be passed
  // to the structs as global extensions.
  uint32_t glfwExtensionCount = 0;
  const char **glfwExtensions;
  glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

  std::vector<const char *> extensions(glfwExtensions,
                                       glfwExtensions + glfwExtensionCount);

  if (enableValidationLayers) {
    // extension needed to add callbacks in the program to handle error
    // (and all kinds of) messages
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }

  return extensions;
}

struct QueueFamilyIndices {
  // there is no value to indicate a non-existent queue family
  // therefore we use optional
  std::optional<uint32_t> graphicsFamily;
  // to store the queue family that supports presenting
  // to the surface we created
  std::optional<uint32_t> presentFamily;

  bool isComplete() {
    return graphicsFamily.has_value() && presentFamily.has_value();
  }
};

struct SwapChainSupportDetails {
  VkSurfaceCapabilitiesKHR capabilities{};
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> presentModes;
};

bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
  uint32_t extensionCount;
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                       nullptr);

  std::vector<VkExtensionProperties> availableExtensions(extensionCount);
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                       availableExtensions.data());

  std::set<std::string> requiredExtensions(deviceExtensions.begin(),
                                           deviceExtensions.end());

  for (const auto &extension : availableExtensions) {
    requiredExtensions.erase(extension.extensionName);
  }

  return requiredExtensions.empty();
}

VkResult CreateDebugUtilsMessengerEXT(
    VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
    const VkAllocationCallbacks *pAllocator,
    VkDebugUtilsMessengerEXT *pDebugMessenger) {
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(
      instance, "vkCreateDebugUtilsMessengerEXT");
  if (func != nullptr) {
    return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
  } else {
    return VK_ERROR_EXTENSION_NOT_PRESENT;
  }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance,
                                   VkDebugUtilsMessengerEXT debugMessenger,
                                   const VkAllocationCallbacks *pAllocator) {
  auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(
      instance, "vkDestroyDebugUtilsMessengerEXT");
  if (func != nullptr) {
    func(instance, debugMessenger, pAllocator);
  }
}

// Class to store the Vulkan objects as members
class HelloTriangleApplication {
 public:
  void run() {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
  }

 private:
  void initWindow() {
    glfwInit();

    // Avoid creating an OpenGL context
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    // Handling resizable windows takes some special care
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    window_ = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
  }

  // initiate vulkan objects
  void initVulkan() {
    if (glfwVulkanSupported()) {
      std::cout << "Vulkan Supported" << std::endl;
      createInstance();
      setupDebugMessenger();
      createSurface();
      pickPhysicalDevice();
      createLogicalDevice();
      createSwapChain();
      createImageViews();
      createRenderPass();
      createGraphicsPipeline();
      createFramebuffers();
      createCommandPool();
      createCommandBuffers();
      createSyncObjects();
    }
  }

  void createSyncObjects() {
    imageAvailableSemaphores_.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores_.resize(MAX_FRAMES_IN_FLIGHT);
    inFlightFences_.resize(MAX_FRAMES_IN_FLIGHT);
    imagesInFlight_.resize(swapChainImages_.size(), VK_NULL_HANDLE);

    // in current API there are no required fields except sType
    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      if (vkCreateSemaphore(device_, &semaphoreInfo, nullptr, &imageAvailableSemaphores_[i]) != VK_SUCCESS ||
          vkCreateSemaphore(device_, &semaphoreInfo, nullptr, &renderFinishedSemaphores_[i]) != VK_SUCCESS ||
          vkCreateFence(device_, &fenceInfo, nullptr, &inFlightFences_[i]) != VK_SUCCESS) {

        throw std::runtime_error("failed to create synchronization objects for a frame!");
      }
    }

  }

  void createCommandBuffers() {
    commandBuffers_.resize(swapChainFramebuffers_.size());
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool_;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = (uint32_t) commandBuffers_.size();

    if (vkAllocateCommandBuffers(device_, &allocInfo, commandBuffers_.data()) != VK_SUCCESS) {
      throw std::runtime_error("failed to allocate command buffers!");
    }

    for (size_t i = 0; i < commandBuffers_.size(); i++) {
      VkCommandBufferBeginInfo beginInfo{};
      beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      beginInfo.flags = 0; // Optional
      beginInfo.pInheritanceInfo = nullptr; // Optional

      if (vkBeginCommandBuffer(commandBuffers_[i], &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("failed to begin recording command buffer!");
      }

      VkRenderPassBeginInfo renderPassInfo{};
      renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
      renderPassInfo.renderPass = renderPass_;
      renderPassInfo.framebuffer = swapChainFramebuffers_[i];

      renderPassInfo.renderArea.offset = {0, 0};
      renderPassInfo.renderArea.extent = swapChainExtent_;

      VkClearValue clearColor = {0.0f, 0.0f, 0.0f, 1.0f};
      renderPassInfo.clearValueCount = 1;
      renderPassInfo.pClearValues = &clearColor;

      vkCmdBeginRenderPass(commandBuffers_[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
      vkCmdBindPipeline(commandBuffers_[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline_);
      vkCmdDraw(commandBuffers_[i], 3, 1, 0, 0);

      vkCmdEndRenderPass(commandBuffers_[i]);

      if (vkEndCommandBuffer(commandBuffers_[i]) != VK_SUCCESS) {
        throw std::runtime_error("failed to record command buffer!");
      }
    }
  }

  void createCommandPool() {
    QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice_);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    // we are going to record commands for drawing, which is why we've
    // chosen the graphics queue family
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
    poolInfo.flags = 0;

    if (vkCreateCommandPool(device_, &poolInfo, nullptr, &commandPool_) != VK_SUCCESS) {
      throw std::runtime_error("failed to create command pool!");
    }
  }

  void createFramebuffers() {
    swapChainFramebuffers_.resize(swapChainImageViews_.size());

    for (size_t i = 0; i < swapChainImageViews_.size(); i++) {
      VkImageView attachments[] = {
          swapChainImageViews_[i]
      };

      VkFramebufferCreateInfo framebufferInfo{};
      framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
      framebufferInfo.renderPass = renderPass_;
      framebufferInfo.attachmentCount = 1;
      framebufferInfo.pAttachments = attachments;
      framebufferInfo.width = swapChainExtent_.width;
      framebufferInfo.height = swapChainExtent_.height;
      framebufferInfo.layers = 1;

      if (vkCreateFramebuffer(device_, &framebufferInfo, nullptr, &swapChainFramebuffers_[i]) != VK_SUCCESS) {
        throw std::runtime_error("failed to create framebuffer!");
      }
    }

  }

  void createRenderPass() {
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = swapChainImageFormat_;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;
    if (vkCreateRenderPass(device_, &renderPassInfo, nullptr, &renderPass_) != VK_SUCCESS) {
      throw std::runtime_error("failed to create render pass!");
    }
  }

  void createGraphicsPipeline() {
    auto vertShaderCode = readFile("shader.vert.spv");
    auto fragShaderCode = readFile("shader.frag.spv");

    VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
    VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertShaderModule;
    // entrypoint function invoked by the pipeline
    vertShaderStageInfo.pName = "main";
    // it's possible to use pSpecializationInfo for constants that can change
    // the behaviour of the shader, though here we set it to nullptr (default)

    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

    // Vertex input
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 0;
    vertexInputInfo.pVertexBindingDescriptions = nullptr; // Optional
    vertexInputInfo.vertexAttributeDescriptionCount = 0;
    vertexInputInfo.pVertexAttributeDescriptions = nullptr; // Optional

    // Input Assembly
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    // topology sounds cool, need to check it out later when drawing
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    // Viewports: region of the framebuffer that the output will be rendered to
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float) swapChainExtent_.width;
    viewport.height = (float) swapChainExtent_.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    // Scissors: filter that tells where rasterizer will actually retain pixels
    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = swapChainExtent_;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    // Rasterizer
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    // Any other mode than fill requires a GPU feature
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    // Any line thicker requires the widelines GPU feature
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;
    rasterizer.depthBiasConstantFactor = 0.0f; // Optional
    rasterizer.depthBiasClamp = 0.0f; // Optional
    rasterizer.depthBiasSlopeFactor = 0.0f; // Optional

    // Multisampling
    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling.minSampleShading = 1.0f; // Optional
    multisampling.pSampleMask = nullptr; // Optional
    multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
    multisampling.alphaToOneEnable = VK_FALSE; // Optional

    // Colour Blending
    // Per framebuffer
    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; // Optional
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; // Optional
    // Global
    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f; // Optional
    colorBlending.blendConstants[1] = 0.0f; // Optional
    colorBlending.blendConstants[2] = 0.0f; // Optional
    colorBlending.blendConstants[3] = 0.0f; // Optional

    // Pipeline State
    // Won't be using this for now, but still have to create an empty state
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 0; // Optional
    pipelineLayoutInfo.pSetLayouts = nullptr; // Optional
    pipelineLayoutInfo.pushConstantRangeCount = 0; // Optional
    pipelineLayoutInfo.pPushConstantRanges = nullptr; // Optional

    if (vkCreatePipelineLayout(device_, &pipelineLayoutInfo, nullptr, &pipelineLayout_) != VK_SUCCESS) {
      throw std::runtime_error("failed to create pipeline layout!");
    }

    // Pipeline Creation
    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;

    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = nullptr; // Optional
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = nullptr; // Optional

    pipelineInfo.layout = pipelineLayout_;

    pipelineInfo.renderPass = renderPass_;
    pipelineInfo.subpass = 0;

    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
    pipelineInfo.basePipelineIndex = -1; // Optional

    if (vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline_) !=
    VK_SUCCESS) {
      throw std::runtime_error("failed to create graphics pipeline!");
    }
    vkDestroyShaderModule(device_, fragShaderModule, nullptr);
    vkDestroyShaderModule(device_, vertShaderModule, nullptr);
  }

  VkShaderModule createShaderModule(const std::vector<char>& code) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    // size of the bytecode is specified in bytes, but the bytecode pointer
    // is a uint32_t pointer rather than a char pointer
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device_, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
      throw std::runtime_error("failed to create shader module!");
    }

    return shaderModule;
  }

  void createImageViews() {
    swapChainImageViews_.resize(swapChainImages_.size());
    for (size_t i = 0; i < swapChainImages_.size(); i++) {
      VkImageViewCreateInfo createInfo{};
      createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      createInfo.image = swapChainImages_[i];

      // viewType allows to treat images as 1D/2D/3D textures and cube maps
      createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
      createInfo.format = swapChainImageFormat_;

      // identity here means id operation a -> a
      createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
      createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
      createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
      createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

      // describes image's purpose and which part of the image should be accessed
      createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      createInfo.subresourceRange.baseMipLevel = 0;
      createInfo.subresourceRange.levelCount = 1;
      createInfo.subresourceRange.baseArrayLayer = 0;
      createInfo.subresourceRange.layerCount = 1;

      // stereographic 3D application would need multiple layers in the swap
      // chain and multiple images views for each image (left/right eyes)

      if (vkCreateImageView(device_, &createInfo, nullptr, &swapChainImageViews_[i]) != VK_SUCCESS) {
        throw std::runtime_error("failed to create image views!");
      }
    }
  }

  void createSwapChain() {
    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice_);

    VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
    VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
    VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

    if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
      imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface_;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    QueueFamilyIndices indices = findQueueFamilies(physicalDevice_);
    uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};

    if (indices.graphicsFamily != indices.presentFamily) {
      // images used across multiple queue families without explicit
      // ownership trasfers; currently avoiding ownership concepts
      // requires at least two distinct queue families
      createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
      createInfo.queueFamilyIndexCount = 2;
      createInfo.pQueueFamilyIndices = queueFamilyIndices;
    } else {
      // image is owned by one queue family at a time and the ownership must
      // be explicitly transferred before using it; offers the best performance
      // in most hardware graphics queue family and presentation queue family
      // are the same, we should stick with exclusive mode then
      createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
      createInfo.queueFamilyIndexCount = 0; // Optional
      createInfo.pQueueFamilyIndices = nullptr; // Optional
    }

    // some certain transform can be specified to apply to all images in
    // the swap chain if it's supported, like rotation or flip
    // to leave it as is, set to current transformation
    createInfo.preTransform = swapChainSupport.capabilities.currentTransform;

    // for overlap with other windows
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

    // we don't care about pixels we don't see (e.g. they are covered with other
    // windows)
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;

    // sometimes swapchain needs to be recreated (e.g. when the window was
    // resized); for now creating only one swap chain
    createInfo.oldSwapchain = VK_NULL_HANDLE;

    if (vkCreateSwapchainKHR(device_, &createInfo, nullptr, &swapChain_) != VK_SUCCESS) {
      throw std::runtime_error("failed to create swap chain!");
    }

    vkGetSwapchainImagesKHR(device_, swapChain_, &imageCount, nullptr);
    swapChainImages_.resize(imageCount);
    vkGetSwapchainImagesKHR(device_, swapChain_, &imageCount, swapChainImages_.data());

    swapChainImageFormat_ = surfaceFormat.format;
    swapChainExtent_ = extent;
  }

  void createSurface() {
    // passed the VkResult from the relevant platform call
    if (glfwCreateWindowSurface(instance_, window_, nullptr, &surface_) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create window surface");
    }
  }

  void createLogicalDevice() {
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice_);

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(),
                                              indices.presentFamily.value()};

    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
      VkDeviceQueueCreateInfo queueCreateInfo{};
      queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      queueCreateInfo.queueFamilyIndex = queueFamily;
      queueCreateInfo.queueCount = 1;
      queueCreateInfo.pQueuePriorities = &queuePriority;
      queueCreateInfos.push_back(queueCreateInfo);
    }

    // we don't need anything special for now,
    // so everything can be left to VK_FALSE
    VkPhysicalDeviceFeatures deviceFeatures{};

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.queueCreateInfoCount =
        static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos = queueCreateInfos.data();

    createInfo.pEnabledFeatures = &deviceFeatures;

    createInfo.enabledExtensionCount =
        static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();

    if (enableValidationLayers) {
      // this field will be ignored by newer implementations
      // as those are propagated from instance createInfo
      // setting them for older versions
      createInfo.enabledLayerCount =
          static_cast<uint32_t>(validationLayers.size());
      createInfo.ppEnabledLayerNames = validationLayers.data();
    } else {
      createInfo.enabledLayerCount = 0;
    }

    if (vkCreateDevice(physicalDevice_, &createInfo, nullptr, &device_) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create logical device");
    }
    vkGetDeviceQueue(device_, indices.graphicsFamily.value(), 0,
                     &graphicsQueue_);
  }

  void pickPhysicalDevice() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr);

    if (deviceCount == 0) {
      throw std::runtime_error("failed to find GPUs with Vulkan support");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data());

    for (const auto &device : devices) {
      if (isDeviceSuitable(device)) {
        physicalDevice_ = device;
        break;
      }
    }

    if (physicalDevice_ == VK_NULL_HANDLE) {
      throw std::runtime_error("no suitable GPU found");
    }
  }

  // A lot of information in Vulkan is passed through structs instead
  // of function parameters.
  void createInstance() {
    if (enableValidationLayers && !checkValidationLayerSupport()) {
      throw std::runtime_error(
          "validation layers requested, but no available!");
    }
    // structure for providing some useful information to the driver
    // in order to optimize our specific applications.
    // it also contains pNext member that can point to extension
    // information in the future, initialized as nullptr at {}
    VkApplicationInfo appInfo{};
    // required explicit sType member
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Hello Triangle";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    // this structs will tell the Vulkan driver which global (for the
    // entire program) extensions and validation layers we want to use.
    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    // retrieving all supported extensions
    uint32_t extensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> extensions(extensionCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());
    // std::cout << "available extensions:\n";
    std::unordered_set<std::string> availableExtensionsNames;
    for (const auto &extension : extensions) {
      // std::cout << '\t' << extension.extensionName << std::endl;
      availableExtensionsNames.insert(extension.extensionName);
    }

    auto glfwExtensions = getRequiredExtensions();

    // std::cout << "required extensions:" << std::endl;
    for (const char *extension : glfwExtensions) {
      // std::cout << '\t' << extension << ' ';
      if (availableExtensionsNames.count(extension)) {
        // std::cout << "available" << std::endl;
      } else {
        // std::cout << "unavailable" << std::endl;
        throw std::runtime_error("Required extension is not available");
      }
    }

    createInfo.enabledExtensionCount =
        static_cast<uint32_t>(glfwExtensions.size());
    createInfo.ppEnabledExtensionNames = glfwExtensions.data();

    // Moved outside of the if statement to ensure that it's not destroyed
    // before vkCreateInstance call.
    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo;
    // determine the global validation layers to enable.
    if (enableValidationLayers) {
      createInfo.enabledLayerCount =
          static_cast<uint32_t>(validationLayers.size());
      createInfo.ppEnabledLayerNames = validationLayers.data();

      populateDebugMessengerCreateInfo(debugCreateInfo);

      createInfo.pNext =
          (VkDebugUtilsMessengerCreateInfoEXT *) &debugCreateInfo;
    } else {
      createInfo.enabledLayerCount = 0;
      createInfo.pNext = nullptr;
    }

    // general pattern of the object creation function parameters
    // in Vulkan
    // pointer with creation    info
    // pointer to custom allocator callbacks
    // pointer that stores the handle to the new object
    // VkResult is either VK_SUCCESS or an error code
    // VkResult result = vkCreateInstance(&createInfo, nullptr, &instance_);
    // if (result != VK_SUCCESS) {
    if (vkCreateInstance(&createInfo, nullptr, &instance_) != VK_SUCCESS) {
      throw std::runtime_error("failed to create instance!");
    } else {
      std::cout << "Instance for application " << appInfo.pApplicationName
                << " created successfully" << std::endl;
    }
  }

  void setupDebugMessenger() {
    if (!enableValidationLayers) {
      return;
    }
    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    populateDebugMessengerCreateInfo(createInfo);

    if (CreateDebugUtilsMessengerEXT(instance_, &createInfo, nullptr,
                                     &debugMessenger_) != VK_SUCCESS) {
      throw std::runtime_error("failed to set up debug messenger!");
    }
  }

  SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
    SwapChainSupportDetails details;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface_,
                                              &details.capabilities);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_, &formatCount, nullptr);

    if (formatCount != 0) {
      details.formats.resize(formatCount);
      vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_, &formatCount, details.formats.data());
    }

    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface_, &presentModeCount, nullptr);

    if (presentModeCount != 0) {
      details.presentModes.resize(presentModeCount);
      vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface_, &presentModeCount, details.presentModes.data());
    }

    return details;
  }

  static VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR> &availablePresentModes) {
    for (const auto &availablePresentMode : availablePresentModes) {
      if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
        return availablePresentMode;
      }
    }
    return VK_PRESENT_MODE_FIFO_KHR;
  }

  static VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &availableFormats) {
    for (const auto &availableFormat : availableFormats) {
      if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB
          && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
        return availableFormat;
      }
    }
    return availableFormats[0];
  }

  static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
      throw std::runtime_error("failed to open file!");
    }

    size_t fileSize = (size_t) file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();

    return buffer;
  }

  VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities) {
    if (capabilities.currentExtent.width != UINT32_MAX) {
      return capabilities.currentExtent;
    } else {
      int width, height;
      glfwGetFramebufferSize(window_, &width, &height);

      VkExtent2D actualExtent = {
          static_cast<uint32_t>(width),
          static_cast<uint32_t>(height)
      };

      actualExtent.width =
          std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
      actualExtent.height = std::max(capabilities.minImageExtent.height,
                                     std::min(capabilities.maxImageExtent.height, actualExtent.height));

      return actualExtent;
    }
  }

  // start rendering frames
  void mainLoop() {
    // until either an error occurs or the window is closed
    while (!glfwWindowShouldClose(window_)) {
      glfwPollEvents();
      drawFrame();
    }
    vkDeviceWaitIdle(device_);
  }

  // - acquire an image from the swap chain
  // - execute the command buffer with that image as attachment in the framebuffer
  // - return the image to the swap chain for presentation
  // these events are executed asynchronously
  void drawFrame() {
    vkWaitForFences(device_, 1, &inFlightFences_[currentFrame_], VK_TRUE, UINT64_MAX);

    uint32_t imageIndex;
    // UINT64_MAX for timeout value disables it
    vkAcquireNextImageKHR(device_, swapChain_, UINT64_MAX,
        imageAvailableSemaphores_[currentFrame_], VK_NULL_HANDLE, &imageIndex);

    if (imagesInFlight_[imageIndex] != VK_NULL_HANDLE) {
      vkWaitForFences(device_, 1, &imagesInFlight_[imageIndex], VK_TRUE, UINT64_MAX);
    }
    imagesInFlight_[imageIndex] = inFlightFences_[currentFrame_];

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkSemaphore waitSemaphores[] = {imageAvailableSemaphores_[currentFrame_]};
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers_[imageIndex];

    VkSemaphore signalSemaphores[] = {renderFinishedSemaphores_[currentFrame_]};
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    vkResetFences(device_, 1, &inFlightFences_[currentFrame_]);

    if (vkQueueSubmit(graphicsQueue_, 1, &submitInfo, inFlightFences_[currentFrame_]) != VK_SUCCESS) {
      throw std::runtime_error("failed to submit draw command buffer!");
    }

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;

    VkSwapchainKHR swapChains[] = {swapChain_};
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;
    presentInfo.pResults = nullptr; // Optional

    // we will add the error handling here later
    vkQueuePresentKHR(presentQueue_, &presentInfo);

    vkQueueWaitIdle(presentQueue_);
    currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES_IN_FLIGHT;
  }

  // once the window is closed
  void cleanup() {
    if (enableValidationLayers) {
      DestroyDebugUtilsMessengerEXT(instance_, debugMessenger_, nullptr);
    }

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      vkDestroySemaphore(device_, renderFinishedSemaphores_[i], nullptr);
      vkDestroySemaphore(device_, imageAvailableSemaphores_[i], nullptr);
      vkDestroyFence(device_, inFlightFences_[i], nullptr);
    }

    vkDestroyCommandPool(device_, commandPool_, nullptr);

    for (auto framebuffer : swapChainFramebuffers_) {
      vkDestroyFramebuffer(device_, framebuffer, nullptr);
    }

    vkDestroyPipeline(device_, graphicsPipeline_, nullptr);

    vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);

    vkDestroyRenderPass(device_, renderPass_, nullptr);

    for (auto imageView : swapChainImageViews_) {
      vkDestroyImageView(device_, imageView, nullptr);
    }

    vkDestroySwapchainKHR(device_, swapChain_, nullptr);

    vkDestroyDevice(device_, nullptr);

    vkDestroySurfaceKHR(instance_, surface_, nullptr);

    vkDestroyInstance(instance_, nullptr);

    glfwDestroyWindow(window_);

    glfwTerminate();
  }

  bool isDeviceSuitable(VkPhysicalDevice device) {
    // VkPhysicalDeviceProperties deviceProperties;
    // vkGetPhysicalDeviceProperties(device, &deviceProperties);
    // VkPhysicalDeviceFeatures deviceFeatures;
    // vkGetPhysicalDeviceFeatures(device, &deviceFeatures);
    // return deviceProperties.deviceType ==
    // VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && deviceFeatures.geometryShader;
    // For some reason geometryShader is not true for both of my cards
    QueueFamilyIndices indices = findQueueFamilies(device);

    bool extensionsSupported = checkDeviceExtensionSupport(device);

    bool swapChainAdequate = false;
    if (extensionsSupported) {
      SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
      swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
    }

    return indices.isComplete() && swapChainAdequate && extensionsSupported;
  }

  QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
    QueueFamilyIndices indices;

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                             nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                             queueFamilies.data());

    int i = 0;
    for (const auto &queueFamily : queueFamilies) {
      if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
        indices.graphicsFamily = i;
      }

      VkBool32 presentSupport = false;
      vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface_,
                                           &presentSupport);

      if (presentSupport) {
        indices.presentFamily = i;
      }

      if (indices.isComplete()) {
        break;
      }

      i++;
    }
    return indices;
  }

  // returns a boolean indicating if the call that triggered the validation
  // layer should be aborted
  static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
      // might be diagnostic/informational/possible_bug/error
      VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
      // unrelated to the specification or performace/something that violates
      // the specification/potential non-optimal use of Vulkan
      VkDebugUtilsMessageTypeFlagsEXT messageType,
      // details of the message itself: message + array of related object
      // handles
      const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
      // something to pass your own data to it
      void *pUserData) {
    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

    return VK_FALSE;
  }

  static void populateDebugMessengerCreateInfo(
      VkDebugUtilsMessengerCreateInfoEXT &createInfo) {
    createInfo = {};
    createInfo.sType =
        VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;
  }

  GLFWwindow *window_;
  // connection between application and Vulkan library
  VkInstance instance_;
  VkDebugUtilsMessengerEXT debugMessenger_;
  VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
  // a logical device to interface with the physical device
  VkDevice device_;
  // handle to interface with the queues which are created along with the
  // logical device
  VkQueue graphicsQueue_;
  // won't be platform agnostic, thankfully it can be created with
  // glfwCreateWindowSurface that can handle the platform differences
  VkSurfaceKHR surface_;
  VkQueue presentQueue_;
  VkSwapchainKHR swapChain_;
  std::vector<VkImage> swapChainImages_;
  VkFormat swapChainImageFormat_;
  VkExtent2D swapChainExtent_;
  std::vector<VkImageView> swapChainImageViews_;
  VkRenderPass renderPass_;
  VkPipelineLayout pipelineLayout_;
  VkPipeline graphicsPipeline_;
  std::vector<VkFramebuffer> swapChainFramebuffers_;
  // command pools manage the memory that is used to store the buffers
  // and command buffers are allocated from them.
  VkCommandPool commandPool_;
  std::vector<VkCommandBuffer> commandBuffers_;
  // for synchronizing the queue operations of draw commands and presentation
  std::vector<VkSemaphore> imageAvailableSemaphores_;
  std::vector<VkSemaphore> renderFinishedSemaphores_;
  std::vector<VkFence> inFlightFences_;
  std::vector<VkFence> imagesInFlight_;
  size_t currentFrame_ = 0;
};

int main() {
  HelloTriangleApplication app;

  try {
    app.run();
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
