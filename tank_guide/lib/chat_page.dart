import 'dart:async';
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

  final String baseUrl = "https://ai-tank-api-13cc.onrender.com";

  List<Map<String, dynamic>> messages = [];
  bool isLoading = false;
  bool serverReady = false;

  @override
  void initState() {
    super.initState();
    wakeServer();
  }

  void scrollToBottom() {
    Future.delayed(const Duration(milliseconds: 100), () {
      if (!_scrollController.hasClients) return;
      _scrollController.animateTo(
        0,
        duration: const Duration(milliseconds: 300),
        curve: Curves.easeOut,
      );
    });
  }

  Future<void> wakeServer() async {
    try {
      final response = await html.HttpRequest.request(
        "$baseUrl/test",
        method: "GET",
      ).timeout(const Duration(seconds: 60));

      print("WAKE STATUS: ${response.status}");
      print("WAKE BODY: ${response.responseText}");

      if (mounted) {
        setState(() {
          serverReady = response.status == 200;
        });
      }
    } catch (e) {
      print("WAKE ERROR: $e");
      if (mounted) {
        setState(() {
          serverReady = false;
        });
      }
    }
  }

  Future<void> sendMessage() async {
    if (controller.text.trim().isEmpty || isLoading) return;

    final text = controller.text.trim();

    setState(() {
      messages.add({"role": "user", "text": text});
      isLoading = true;
    });

    scrollToBottom();

    try {
      if (!serverReady) {
        await wakeServer();
      }

      final response = await html.HttpRequest.request(
        "$baseUrl/chat",
        method: "POST",
        sendData: jsonEncode({"message": text}),
        requestHeaders: {
          "Content-Type": "application/json",
        },
      ).timeout(const Duration(seconds: 90));

      print("CHAT STATUS: ${response.status}");
      print("CHAT BODY: ${response.responseText}");

      if (response.status != 200) {
        throw Exception(
          "Chat failed. Status: ${response.status}, Body: ${response.responseText}",
        );
      }

      final decoded = jsonDecode(response.responseText!);

      setState(() {
        messages.add({
          "role": "bot",
          "text": decoded["reply"] ?? "No reply from server",
        });
        isLoading = false;
      });
    } on TimeoutException {
      setState(() {
        messages.add({
          "role": "bot",
          "text":
              "Server took too long. Render free tier may be sleeping. Try again."
        });
        isLoading = false;
      });
    } catch (e, st) {
      print("CHAT ERROR: $e");
      print(st);

      setState(() {
        messages.add({
          "role": "bot",
          "text": "Chat failed: $e",
        });
        isLoading = false;
      });
    }

    controller.clear();
    scrollToBottom();
  }

  Future<void> sendImage() async {
    if (isLoading) return;

    final input = html.FileUploadInputElement();
    input.accept = 'image/*';
    input.click();

    input.onChange.listen((event) {
      if (input.files == null || input.files!.isEmpty) return;

      final file = input.files!.first;
      final reader = html.FileReader();

      reader.readAsArrayBuffer(file);

      reader.onLoadEnd.listen((event) async {
        final result = reader.result;
        if (result == null) return;

        Uint8List bytes;
        if (result is Uint8List) {
          bytes = result;
        } else if (result is ByteBuffer) {
          bytes = Uint8List.view(result);
        } else {
          bytes = Uint8List.fromList(result as List<int>);
        }

        setState(() {
          messages.add({"role": "user", "image": bytes});
          isLoading = true;
        });

        scrollToBottom();

        final formData = html.FormData();
        formData.appendBlob("file", html.Blob([bytes]), file.name);

        try {
          if (!serverReady) {
            await wakeServer();
          }

          final response = await html.HttpRequest.request(
            "$baseUrl/predict",
            method: "POST",
            sendData: formData,
          ).timeout(const Duration(seconds: 120));

          print("PREDICT STATUS: ${response.status}");
          print("PREDICT BODY: ${response.responseText}");

          if (response.status != 200) {
            throw Exception(
              "Predict failed. Status: ${response.status}, Body: ${response.responseText}",
            );
          }

          final decoded = jsonDecode(response.responseText!);

          setState(() {
            messages.add({
              "role": "bot",
              "text": decoded["error"] ??
                  "Tank: ${decoded['tank_name']}\n"
                      "Confidence: ${((decoded['confidence'] ?? 0) * 100).toStringAsFixed(2)}%\n\n"
                      "${decoded['gpt_explanation'] ?? decoded['description'] ?? 'No explanation'}",
            });
            isLoading = false;
          });
        } on TimeoutException {
          setState(() {
            messages.add({
              "role": "bot",
              "text":
                  "Image request timed out. Render free tier may be sleeping. Try again."
            });
            isLoading = false;
          });
        } catch (e, st) {
          print("IMAGE ERROR: $e");
          print(st);

          setState(() {
            messages.add({
              "role": "bot",
              "text": "Failed to send image: $e",
            });
            isLoading = false;
          });
        }

        scrollToBottom();
      });

      reader.onError.listen((event) {
        setState(() {
          messages.add({
            "role": "bot",
            "text": "Failed to read the selected image.",
          });
          isLoading = false;
        });
      });
    });
  }

  Widget buildMessage(Map<String, dynamic> msg) {
    final bool isUser = msg['role'] == 'user';

    return Container(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      margin: const EdgeInsets.all(8),
      child: Container(
        constraints: const BoxConstraints(maxWidth: 260),
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: isUser ? Colors.blue : Colors.grey[300],
          borderRadius: BorderRadius.circular(10),
        ),
        child: msg["image"] != null
            ? Image.memory(msg["image"], height: 150)
            : Text(
                msg["text"] ?? "",
                style: TextStyle(
                  color: isUser ? Colors.white : Colors.black,
                ),
              ),
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
            icon: Icon(
              serverReady ? Icons.cloud_done : Icons.cloud_off,
            ),
          ),
        ],
      ),
      body: Column(
        children: [
          if (!serverReady)
            Container(
              width: double.infinity,
              color: Colors.amber.shade200,
              padding: const EdgeInsets.all(8),
              child: const Text(
                "Waking server... first request may be slow on Render free tier.",
                textAlign: TextAlign.center,
              ),
            ),
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
          if (isLoading)
            const Padding(
              padding: EdgeInsets.all(8.0),
              child: CircularProgressIndicator(),
            ),
          Padding(
            padding: const EdgeInsets.all(10),
            child: Row(
              children: [
                IconButton(
                  icon: const Icon(Icons.image),
                  onPressed: isLoading ? null : sendImage,
                ),
                Expanded(
                  child: TextField(
                    controller: controller,
                    onSubmitted: (_) => sendMessage(),
                    decoration: const InputDecoration(
                      hintText: "Ask about tank...",
                      border: OutlineInputBorder(),
                    ),
                  ),
                ),
                IconButton(
                  icon: const Icon(Icons.send),
                  onPressed: isLoading ? null : sendMessage,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}