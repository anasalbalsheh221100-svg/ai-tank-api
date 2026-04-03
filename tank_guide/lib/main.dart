import 'package:flutter/material.dart';
import 'chat_page.dart';

void main() {
  runApp(const TankGuideApp());
}

class TankGuideApp extends StatelessWidget {
  const TankGuideApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Tank Guide',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: const ChatPage(),
    );
  }
}