
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:http_parser/http_parser.dart';

class ChatPage extends StatefulWidget {
  const ChatPage({super.key});

  @override
  State<ChatPage> createState() => _ChatPageState();
}

class _ChatPageState extends State<ChatPage> {
  final TextEditingController controller = TextEditingController();
  final picker = ImagePicker();
  final ScrollController _scrollController = ScrollController();

  List<Map<String, dynamic>> messages = [];
  bool isLoading = false;

  // =========================
  // AUTO SCROLL
  // =========================
  void scrollToBottom() {
    Future.delayed(const Duration(milliseconds: 100), () {
      _scrollController.animateTo(
        0,
        duration: const Duration(milliseconds: 300),
        curve: Curves.easeOut,
      );
    });
  }

  // =========================
  // SEND TEXT
  // =========================
  Future<void> sendMessage() async {
    if (controller.text.isEmpty) return;

    final userText = controller.text;

    setState(() {
      messages.add({"role": "user", "text": userText});
      isLoading = true;
    });

    scrollToBottom();

    try {
      var res = await http.post(
        Uri.parse("https://ai-tank-api-13cc.onrender.com/chat"), // ✅ updated URL
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({"message": userText}),
      ).timeout(const Duration(seconds: 60));

      if (res.statusCode != 200) {
        setState(() {
          messages.add({
            "role": "bot",
            "text": "Server error: ${res.body}",
          });
          isLoading = false;
        });
        return;
      }

      final decoded = jsonDecode(res.body);

      setState(() {
        messages.add({
          "role": "bot",
          "text": decoded['reply'] ?? "No reply"
        });
        isLoading = false;
      });
    } catch (e) {
      setState(() {
        messages.add({
          "role": "bot",
          "text": "Error sending message"
        });
        isLoading = false;
      });
    }

    scrollToBottom();
    controller.clear();
  }

  // =========================
  // SEND IMAGE
  // =========================
  Future<void> sendImage() async {
    final picked = await picker.pickImage(source: ImageSource.gallery);
    if (picked == null) return;

    Uint8List bytes = await picked.readAsBytes();

    setState(() {
      messages.add({"role": "user", "image": bytes});
      isLoading = true;
    });

    scrollToBottom();

    var request = http.MultipartRequest(
      'POST',
      Uri.parse("https://ai-tank-api-13cc.onrender.com/predict"), // ✅ updated URL
    );

    request.files.add(
      http.MultipartFile.fromBytes(
        'file',
        bytes,
        filename: "image.jpg",
        contentType: MediaType('image', 'jpeg'),
      ),
    );

    try {
      var response = await request.send().timeout(
        const Duration(seconds: 60),
      );

      var resBody = await response.stream.bytesToString();
      final decoded = jsonDecode(resBody);

      setState(() {
        messages.add({
          "role": "bot",
          "text": decoded["error"] ??
              "Tank: ${decoded['tank_name']}\n"
              "Confidence: ${(decoded['confidence'] * 100).toStringAsFixed(2)}%\n\n"
              "${decoded['gpt_explanation']}", // ✅ added GPT output
        });
        isLoading = false;
      });
    } catch (e) {
        print("ERROR: $e");
      setState(() {
        messages.add({
          "role": "bot",
          "text": "Failed to send image",
        });
        isLoading = false;
      });
    }

    scrollToBottom();
  }

  // =========================
  // MESSAGE UI
  // =========================
  Widget buildMessage(Map<String, dynamic> msg) {
    bool isUser = msg['role'] == 'user';

    return Container(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      margin: const EdgeInsets.all(8),
      child: Container(
        constraints: const BoxConstraints(maxWidth: 250),
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: isUser ? Colors.blue : Colors.grey[300],
          borderRadius: BorderRadius.circular(10),
        ),
        child: msg["image"] != null
            ? Image.memory(msg["image"]!, height: 150)
            : Text(
                msg['text'] ?? "",
                style: TextStyle(
                  color: isUser ? Colors.white : Colors.black,
                ),
              ),
      ),
    );
  }

  // =========================
  // UI
  // =========================
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Tank Chat")),
      body: Column(
        children: [
          Expanded(
            child: ListView.builder(
              controller: _scrollController,
              reverse: true,
              itemCount: messages.length,
              itemBuilder: (context, index) {
                final msg = messages[messages.length - 1 - index];
                return buildMessage(msg);
              },
            ),
          ),
          if (isLoading) const CircularProgressIndicator(),
          Padding(
            padding: const EdgeInsets.all(10),
            child: Row(
              children: [
                IconButton(
                  icon: const Icon(Icons.image),
                  onPressed: sendImage,
                ),
                Expanded(
                  child: TextField(
                    controller: controller,
                    decoration: const InputDecoration(
                      hintText: "Ask about tank...",
                      border: OutlineInputBorder(),
                    ),
                  ),
                ),
                IconButton(
                  icon: const Icon(Icons.send),
                  onPressed: sendMessage,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

