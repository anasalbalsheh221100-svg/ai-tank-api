import 'dart:convert';
import 'dart:typed_data';
import 'dart:html' as html;
import 'package:flutter/material.dart';

class ChatPage extends StatefulWidget {
  const ChatPage({super.key});

  @override
  State<ChatPage> createState() => _ChatPageState();
}

class _ChatPageState extends State<ChatPage> {
  final TextEditingController controller = TextEditingController();
  final ScrollController _scrollController = ScrollController();

  List<Map<String, dynamic>> messages = [];
  bool isLoading = false;

  final String baseUrl = "https://ai-tank-api-13cc.onrender.com";

  // =========================
  // SCROLL
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
  // SEND TEXT (FIXED)
  // =========================
  Future<void> sendMessage() async {
    if (controller.text.isEmpty) return;

    final text = controller.text;

    setState(() {
      messages.add({"role": "user", "text": text});
      isLoading = true;
    });

    try {
      final response = await html.HttpRequest.request(
        "$baseUrl/chat",
        method: "POST",
        sendData: jsonEncode({"message": text}),
        requestHeaders: {
          "Content-Type": "application/json",
        },
      );

      print("STATUS: ${response.status}");
      print("BODY: ${response.responseText}");

      final decoded = jsonDecode(response.responseText!);

      setState(() {
        messages.add({
          "role": "bot",
          "text": decoded['reply'] ?? "No reply"
        });
        isLoading = false;
      });
    } catch (e) {
      print("CHAT ERROR: $e");

      setState(() {
        messages.add({
          "role": "bot",
          "text": "Error: $e"
        });
        isLoading = false;
      });
    }

    controller.clear();
    scrollToBottom();
  }

  // =========================
  // SEND IMAGE (FIXED)
  // =========================
  Future<void> sendImage() async {
    final input = html.FileUploadInputElement();
    input.accept = 'image/*';
    input.click();

    input.onChange.listen((event) {
      final file = input.files!.first;
      final reader = html.FileReader();

      reader.readAsArrayBuffer(file);

      reader.onLoadEnd.listen((event) async {
        Uint8List bytes = reader.result as Uint8List;

        setState(() {
          messages.add({"role": "user", "image": bytes});
          isLoading = true;
        });

        final formData = html.FormData();
        formData.appendBlob("file", html.Blob([bytes]), "image.jpg");

        try {
          final response = await html.HttpRequest.request(
            "$baseUrl/predict",
            method: "POST",
            sendData: formData,
          );

          print("PREDICT: ${response.responseText}");

          final decoded = jsonDecode(response.responseText!);

          setState(() {
            messages.add({
              "role": "bot",
              "text": decoded["error"] ??
                  "Tank: ${decoded['tank_name']}\n"
                  "Confidence: ${(decoded['confidence'] * 100).toStringAsFixed(2)}%\n\n"
                  "${decoded['gpt_explanation']}",
            });
            isLoading = false;
          });
        } catch (e) {
          print("IMAGE ERROR: $e");

          setState(() {
            messages.add({
              "role": "bot",
              "text": "Failed to send image"
            });
            isLoading = false;
          });
        }

        scrollToBottom();
      });
    });
  }

  // =========================
  // UI MESSAGE
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