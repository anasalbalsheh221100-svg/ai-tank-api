import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';

class ChatPage extends StatefulWidget {
  const ChatPage({super.key});

  @override
  State<ChatPage> createState() => _ChatPageState();
}

class _ChatPageState extends State<ChatPage> {
  final TextEditingController controller = TextEditingController();
  final ScrollController scrollController = ScrollController();
  final ImagePicker picker = ImagePicker();

  final String baseUrl = "https://ai-tank-api-l3cc.onrender.com";
  // emulator: http://10.0.2.2:8000
  // real phone: http://YOUR_PC_IP:8000

  List<Map<String, dynamic>> messages = [];
  bool isLoading = false;
  bool serverReady = false;

  Uint8List? pendingImageBytes;
  String? pendingImageName;

  @override
  void initState() {
    super.initState();
    wakeServer();
  }

  @override
  void dispose() {
    controller.dispose();
    scrollController.dispose();
    super.dispose();
  }

  void scrollToBottom() {
    Future.delayed(const Duration(milliseconds: 150), () {
      if (!scrollController.hasClients) return;
      scrollController.animateTo(
        0,
        duration: const Duration(milliseconds: 300),
        curve: Curves.easeOut,
      );
    });
  }

  Future<void> wakeServer() async {
    try {
      final response = await http
          .get(Uri.parse("$baseUrl/test"))
          .timeout(const Duration(seconds: 60));

      if (!mounted) return;

      setState(() {
        serverReady = response.statusCode == 200;
      });
    } catch (_) {
      if (!mounted) return;

      setState(() {
        serverReady = false;
      });
    }
  }

  Future<void> pickImage(ImageSource source) async {
    if (isLoading) return;

    try {
      final XFile? pickedFile = await picker.pickImage(source: source);
      if (pickedFile == null) return;

      final Uint8List bytes = await pickedFile.readAsBytes();

      setState(() {
        pendingImageBytes = bytes;
        pendingImageName =
            pickedFile.name.isNotEmpty ? pickedFile.name : "image.jpg";
      });
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Failed to pick image: $e")),
      );
    }
  }

  void removePendingImage() {
    setState(() {
      pendingImageBytes = null;
      pendingImageName = null;
    });
  }

  Future<Map<String, dynamic>> sendPredictRequest(
    Uint8List bytes,
    String filename,
    String message,
  ) async {
    if (!serverReady) {
      await wakeServer();
    }

    final request = http.MultipartRequest(
      "POST",
      Uri.parse("$baseUrl/predict"),
    );

    request.files.add(
      http.MultipartFile.fromBytes(
        "file",
        bytes,
        filename: filename,
      ),
    );

    request.fields["message"] = message;

    final streamedResponse =
        await request.send().timeout(const Duration(seconds: 120));

    final responseBody = await streamedResponse.stream.bytesToString();

    if (streamedResponse.statusCode != 200) {
      throw Exception(
        "Request failed: ${streamedResponse.statusCode}\n$responseBody",
      );
    }

    return jsonDecode(responseBody) as Map<String, dynamic>;
  }

  Future<Map<String, dynamic>> sendChatRequest(String text) async {
    if (!serverReady) {
      await wakeServer();
    }

    final response = await http
        .post(
          Uri.parse("$baseUrl/chat"),
          headers: {"Content-Type": "application/json"},
          body: jsonEncode({"message": text}),
        )
        .timeout(const Duration(seconds: 90));

    if (response.statusCode != 200) {
      throw Exception("Server error: ${response.statusCode}\n${response.body}");
    }

    return jsonDecode(response.body) as Map<String, dynamic>;
  }

  Future<void> sendCombinedMessage() async {
    if (isLoading) return;

    final text = controller.text.trim();
    final hasText = text.isNotEmpty;
    final hasImage = pendingImageBytes != null;

    if (!hasText && !hasImage) return;

    final Uint8List? imageBytes = pendingImageBytes;
    final String imageName = pendingImageName ?? "image.jpg";

    setState(() {
      if (hasImage) {
        messages.add({
          "role": "user",
          "image": imageBytes,
          if (hasText) "text": text,
        });
      } else {
        messages.add({"role": "user", "text": text});
      }

      isLoading = true;
      controller.clear();
      pendingImageBytes = null;
      pendingImageName = null;
    });

    scrollToBottom();

    try {
      if (hasImage && imageBytes != null) {
        final decoded = await sendPredictRequest(imageBytes, imageName, text);

        final replyText = decoded["error"] ??
            "Tank: ${decoded['tank_name']}\n\n"
                "${decoded['reply'] ?? decoded['description'] ?? 'No reply'}";

        setState(() {
          messages.add({
            "role": "bot",
            "text": replyText,
          });
        });
      } else {
        final decoded = await sendChatRequest(text);

        setState(() {
          messages.add({
            "role": "bot",
            "text": decoded["reply"] ?? "No reply from server",
          });
        });
      }
    } on TimeoutException {
      setState(() {
        messages.add({
          "role": "bot",
          "text": "Request timed out. Try again.",
        });
      });
    } catch (e) {
      setState(() {
        messages.add({
          "role": "bot",
          "text": "Failed: $e",
        });
      });
    }

    setState(() {
      isLoading = false;
    });

    scrollToBottom();
  }

  void showImageOptions() {
    showModalBottomSheet(
      context: context,
      builder: (context) {
        return SafeArea(
          child: Wrap(
            children: [
              ListTile(
                leading: const Icon(Icons.camera_alt),
                title: const Text("Camera"),
                onTap: () {
                  Navigator.pop(context);
                  pickImage(ImageSource.camera);
                },
              ),
              ListTile(
                leading: const Icon(Icons.photo),
                title: const Text("Gallery"),
                onTap: () {
                  Navigator.pop(context);
                  pickImage(ImageSource.gallery);
                },
              ),
            ],
          ),
        );
      },
    );
  }

  Widget buildMessage(Map<String, dynamic> msg) {
    final bool isUser = msg["role"] == "user";

    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
        padding: const EdgeInsets.all(12),
        constraints: const BoxConstraints(maxWidth: 300),
        decoration: BoxDecoration(
          color: isUser ? Colors.blue : Colors.grey.shade300,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (msg["image"] != null)
              ClipRRect(
                borderRadius: BorderRadius.circular(8),
                child: Image.memory(
                  msg["image"],
                  height: 160,
                  fit: BoxFit.cover,
                ),
              ),
            if (msg["image"] != null && msg["text"] != null)
              const SizedBox(height: 8),
            if (msg["text"] != null)
              Text(
                msg["text"] ?? "",
                style: TextStyle(
                  color: isUser ? Colors.white : Colors.black,
                  fontSize: 16,
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget buildPendingImagePreview() {
    if (pendingImageBytes == null) return const SizedBox.shrink();

    return Container(
      margin: const EdgeInsets.fromLTRB(10, 0, 10, 8),
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: Colors.grey.shade200,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.grey.shade400),
      ),
      child: Row(
        children: [
          ClipRRect(
            borderRadius: BorderRadius.circular(8),
            child: Image.memory(
              pendingImageBytes!,
              width: 70,
              height: 70,
              fit: BoxFit.cover,
            ),
          ),
          const SizedBox(width: 10),
          const Expanded(
            child: Text(
              "Image selected\nYou can add a message then press Send",
              style: TextStyle(fontSize: 14),
            ),
          ),
          IconButton(
            onPressed: isLoading ? null : removePendingImage,
            icon: const Icon(Icons.close),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Tank Chat"),
        actions: [
          IconButton(
            onPressed: wakeServer,
            icon: Icon(serverReady ? Icons.cloud_done : Icons.cloud_off),
          ),
        ],
      ),
      body: Column(
        children: [
          if (!serverReady)
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(8),
              color: Colors.amber.shade200,
              child: const Text(
                "Waking server... first request may be slow.",
                textAlign: TextAlign.center,
              ),
            ),
          Expanded(
            child: ListView.builder(
              controller: scrollController,
              reverse: true,
              itemCount: messages.length,
              itemBuilder: (context, index) {
                final msg = messages[messages.length - 1 - index];
                return buildMessage(msg);
              },
            ),
          ),
          if (isLoading)
            const Padding(
              padding: EdgeInsets.all(8.0),
              child: CircularProgressIndicator(),
            ),
          buildPendingImagePreview(),
          Padding(
            padding: const EdgeInsets.all(10),
            child: Row(
              children: [
                IconButton(
                  onPressed: isLoading ? null : showImageOptions,
                  icon: const Icon(Icons.image),
                ),
                Expanded(
                  child: TextField(
                    controller: controller,
                    onSubmitted: (_) => sendCombinedMessage(),
                    decoration: const InputDecoration(
                      hintText: "Ask about tank...",
                      border: OutlineInputBorder(),
                    ),
                  ),
                ),
                IconButton(
                  onPressed: isLoading ? null : sendCombinedMessage,
                  icon: const Icon(Icons.send),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}