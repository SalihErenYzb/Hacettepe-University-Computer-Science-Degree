
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import javafx.animation.FadeTransition;
import javafx.animation.Timeline;
import javafx.beans.property.IntegerProperty;
import javafx.beans.property.SimpleIntegerProperty;
import javafx.geometry.Bounds;
import javafx.geometry.Pos;
import javafx.scene.Group;
import javafx.scene.input.KeyCode;
import javafx.scene.layout.BorderPane;
import javafx.scene.media.Media;
import javafx.scene.media.MediaPlayer;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;
import javafx.scene.text.Text;
import javafx.scene.text.TextAlignment;
import javafx.util.Duration;
//This group will give the ability to create any type of level by giving ducks as array list.
public class CustomLevelGroup extends Group {
	private Map<DuckCrossFlyPane, Integer> duckCrossFlyPaneDict = new HashMap<>();
    private Map<DuckVerticalFlyPane, Integer> duckVerticalFlyPaneDict = new HashMap<>();
    private IntegerProperty ammo ;
    private String level;
    private Text rightText;
    private ArrayList<DuckVerticalFlyPane> vert;
    private ArrayList<DuckCrossFlyPane> cross;
    private boolean isOver;
    CustomLevelGroup(ArrayList<DuckVerticalFlyPane> vert,ArrayList<DuckCrossFlyPane> cross,
    		String level, GameStateManager.GameState state){
    	isOver = false;
    	this.level = level+"/6";
        ImagePane backPane = new ImagePane(DuckHunt.Backgrounds[DuckHunt.getI()]);
        ImagePane forePane = new ImagePane(DuckHunt.Foregrounds[DuckHunt.getI()]);
        
        //array lists
        this.cross = cross;
        this.vert = vert;
        for (DuckCrossFlyPane crossDuck: this.cross) {
        	duckCrossFlyPaneDict.put(crossDuck, 1);
        }

        for (DuckVerticalFlyPane verticalDuck: this.vert) {
        	duckVerticalFlyPaneDict.put(verticalDuck, 1);
        }
        this.ammo= new SimpleIntegerProperty(this.getDuckCount()*3);

        //level and ammo texts and their font and layout adjustments
        Text centerText = centerTextCreator();
        rightText = rightTextCreator();


     // create a Media object for the audio file
        String audioFile = "assets/effects/Gunshot.mp3";
        Media media =  new Media(getClass().getResource(audioFile).toExternalForm());

        // create a MediaPlayer object
        MediaPlayer mediaPlayer = new MediaPlayer(media);
        mediaPlayer.setVolume(DuckHunt.VOLUME);

	   // check for cases
        this.setOnMouseClicked(e ->{
        	if (isOver == false){
	        	if (ammo.get() > 0) {
			        double x = e.getX();
			        double y = e.getY();
			        for (DuckVerticalFlyPane verticalDuck: this.vert) {
				        // get the bounds of the ImageView object in the Pane's coordinate system
				        Bounds bounds = verticalDuck.duck.getBoundsInParent();
				        double minX = bounds.getMinX();
				        double maxX = bounds.getMaxX();
				        double minY = bounds.getMinY();
				        double maxY = bounds.getMaxY();
			        
				        // check if the mouse click coordinates are within the bounds of the ImageView object
				        if ((x >= minX) && (x <= maxX) && (y >= minY) && (y <= maxY)) {
				            // handle the mouse click event for the ImageView object
				        	duckVerticalFlyPaneDict.put(verticalDuck, 0);
				        	
				        	verticalDuck.animation2.play();
				        }
			        }
			        for (DuckCrossFlyPane crossDuck: this.cross) {
				        // get the bounds of the ImageView object in the Pane's coordinate system
				        Bounds bounds = crossDuck.duck.getBoundsInParent();
				        double minX = bounds.getMinX();
				        double maxX = bounds.getMaxX();
				        double minY = bounds.getMinY();
				        double maxY = bounds.getMaxY();
			        
				        // check if the mouse click coordinates are within the bounds of the ImageView object
				        if ((x >= minX) && (x <= maxX) && (y >= minY) && (y <= maxY)) {
				            // handle the mouse click event for the ImageView object
				        	
				        	duckCrossFlyPaneDict.put(crossDuck, 0);
				        	crossDuck.animation2.play();
				        }
			        }
	        	}
	            ammo.set(ammo.get() - 1);
	            if ((ammo.get() == 0) && (getDuckCount()!=0)) {
	                // end the game
	            	isOver = true;
	            	for (DuckCrossFlyPane crossDuck: this.cross) {
	            		crossDuck.setMouseTransparent(true);
	                }
	                for (DuckVerticalFlyPane verticalDuck: this.vert) {
	                	verticalDuck.setMouseTransparent(true);
	                }
	            	
	              //media play and control
	    		    String audioFile2 = "assets/effects/GameOver.mp3";
	    	        Media media2 =  new Media(getClass().getResource(audioFile2).toExternalForm());
	    	        MediaPlayer mediaPlayer2 = new MediaPlayer(media2);
	    	        mediaPlayer2.setVolume(DuckHunt.VOLUME);
	
	            	//controls
	            	this.getChildren().add(this.winPaneCreator("GAME OVER!\n\n"));
	        		this.getChildren().add(this.winPaneCreatorSecond("\nPress ENTER to play again\nPress ESC to exit"));
	        		this.requestFocus();
	        		this.setOnKeyPressed(e2 ->{
	        	    	if (e2.getCode() == KeyCode.ENTER) {
	        	    		DuckHunt.Itself().setScene(new GameScene());
	        	    		mediaPlayer.stop();
	        	    		shutDuck();
	        	    		mediaPlayer2.stop();
	        	    	}else if(e2.getCode() == KeyCode.ESCAPE) {
	        	    		mediaPlayer.stop();
	        	    		shutDuck();
	        	    		mediaPlayer2.stop();
	                    	DuckHunt.getGameStateManager()
	        	    		.setCurrentState(GameStateManager.GameState.TITLE);
	        	    	}
	        	    	
	        		});
	        		
	            	
	    		    mediaPlayer2.play();
	    		    mediaPlayer2.setOnEndOfMedia(() -> {
	    	            mediaPlayer2.seek(Duration.ZERO);
	    	            mediaPlayer2.stop();
	    	        });
	            }else if (getDuckCount()==0) {
	            	///This is for level 6
	            	if (level.equals("level 6")) {
	                	//media play and control
	            		isOver = true;
	        		    String audioFile3 = "assets/effects/GameCompleted.mp3";
	        	        Media media3 =  new Media(getClass().getResource(audioFile3).toExternalForm());
	        	        MediaPlayer mediaPlayer3 = new MediaPlayer(media3);
	        	        mediaPlayer3.setVolume(DuckHunt.VOLUME);
	
	        		    mediaPlayer3.play();
	        		    
	            		//add win text
	            		this.getChildren().add(this.winPaneCreator("You have completed the game!\n\n"));
	            		this.getChildren().add(this.winPaneCreatorSecond("\nPress ENTER to play again\nPress ESC to exit"));
	            		this.requestFocus();
	            		this.setOnKeyPressed(e2 ->{
	            	    	if (e2.getCode() == KeyCode.ENTER) {
		        	    		mediaPlayer.stop();
	            	    		mediaPlayer3.stop();
	            	    		shutDuck();
	                	DuckHunt.getGameStateManager()
	    	    		.setCurrentState(state);
	            	    	}else if (e2.getCode() == KeyCode.ESCAPE) {
		        	    		mediaPlayer.stop();
		        	    		shutDuck();
	            	    		mediaPlayer3.stop();
	                        	DuckHunt.getGameStateManager()
	            	    		.setCurrentState(GameStateManager.GameState.TITLE);
	                    	    	}
	            		});
	
	        		    mediaPlayer3.setOnEndOfMedia(() -> {
	        	            mediaPlayer3.stop();
	        	        });
	            	}else {
	            		isOver = true;
		        		//media play and control
		    		    String audioFile3 = "assets/effects/LevelCompleted.mp3";
		    	        Media media3 =  new Media(getClass().getResource(audioFile3).toExternalForm());
		    	        MediaPlayer mediaPlayer3 = new MediaPlayer(media3);
		    	        mediaPlayer3.setVolume(DuckHunt.VOLUME);
	
		            	//add win text
		        		this.getChildren().add(this.winPaneCreator("YOU WIN!\n"));
		        		this.getChildren().add(this.winPaneCreatorSecond("\nPress ENTER to play next level"));
		        		// in case of enter pressed go to level 2
		        		this.requestFocus();
		        		this.setOnKeyPressed(e2 ->{
		        	    	if (e2.getCode() == KeyCode.ENTER) {
		        	    		shutDuck();
		        	    		mediaPlayer.stop();
		        	    		mediaPlayer3.stop();
		            	DuckHunt.getGameStateManager()
			    		.setCurrentState(state);
		        	    	}
		        		});
	
		    		    mediaPlayer3.play();
		    		    mediaPlayer3.setOnEndOfMedia(() -> {
		    	            mediaPlayer3.stop();
		    	        });
	            	}
	            }
	            // play the audio file
	            mediaPlayer.seek(Duration.ZERO);
	            mediaPlayer.play();
        	}
	        });
        	
	        //make sure media is playable after it is played once
	        mediaPlayer.setOnEndOfMedia(() -> {
	            mediaPlayer.seek(Duration.ZERO);
	            mediaPlayer.stop();
	        });
        
        //make sure duck can get killed
        rightText.setMouseTransparent(true);
        centerText.setMouseTransparent(true);
        forePane.setMouseTransparent(true);
        this.getChildren().addAll(backPane);
        this.getChildren().addAll(vert);
        this.getChildren().addAll(cross);
        this.getChildren().addAll(forePane,rightText,centerText);

        // Add a listener to the ammo property
        ammo.addListener((obs, oldVal, newVal) -> {
            rightText.setText(String.format("Ammo Left: %d", newVal));
        });

    }
    //These methods are to create necessery texts
    public Text centerTextCreator() {
    	Text centerText = new Text(level);
        centerText.setLayoutY(10*DuckHunt.SCALE);
        centerText.setLayoutX(240*DuckHunt.SCALE/2-15*DuckHunt.SCALE);
        centerText.setFill(Color.rgb(255, 165, 0)); // Orange color
        centerText.setFont(Font.font("Arial", FontWeight.BOLD, 9*DuckHunt.SCALE));
        return centerText;

    }
    public Text rightTextCreator() {
    	rightText = new Text(String.format("Ammo Left: %d", ammo.get()));
        rightText.setLayoutY(10*DuckHunt.SCALE);
        rightText.setLayoutX(240*DuckHunt.SCALE-50*DuckHunt.SCALE);
        rightText.setTextAlignment(TextAlignment.LEFT);
        rightText.setFill(Color.rgb(255, 165, 0)); // Orange color
        rightText.setFont(Font.font("Arial", FontWeight.BOLD, 9*DuckHunt.SCALE));
        return rightText;
    }
    public BorderPane winPaneCreator(String str) {
		BorderPane borderPane = new BorderPane();
		Text text = new Text(str);
		text.setFont(Font.font("Arial", FontWeight.BOLD, 10*DuckHunt.SCALE));
		text.setTextAlignment(TextAlignment.CENTER);
		text.setFill(Color.rgb(255, 165, 0)); // Orange color
		
		BorderPane.setAlignment(text, Pos.CENTER); // Center the text inside the BorderPane
		borderPane.setPrefSize(DuckHunt.SCALE*256, 240*DuckHunt.SCALE);
		//borderPane.setTop(text);
		borderPane.setCenter(text);
		return borderPane;
    }
    public BorderPane winPaneCreatorSecond(String str) {
		BorderPane borderPane = new BorderPane();
		Text text = new Text(str);
		text.setFont(Font.font("Arial", FontWeight.BOLD, 10*DuckHunt.SCALE));
		text.setTextAlignment(TextAlignment.CENTER);
		text.setFill(Color.rgb(255, 165, 0)); // Orange color
		// create a FadeTransition object
		FadeTransition fadeTransition = new FadeTransition(Duration.seconds(0.5), text);
		fadeTransition.setFromValue(1.0);
		fadeTransition.setToValue(0.0);
		fadeTransition.setCycleCount(Timeline.INDEFINITE);
		fadeTransition.setAutoReverse(true);

		// start the FadeTransition animation
		fadeTransition.play();
		
		BorderPane.setAlignment(text, Pos.CENTER); // Center the text inside the BorderPane
		borderPane.setPrefSize(DuckHunt.SCALE*256, 240*DuckHunt.SCALE);
		//borderPane.setTop(text);
		borderPane.setCenter(text);
		return borderPane;
    }
    public int getDuckCount() {
        int sum = 0;
        for (int value : duckCrossFlyPaneDict.values()) {
            sum += value;
        }
        for (int value : duckVerticalFlyPaneDict.values()) {
            sum += value;
        }
        return sum;
    }
    public void shutDuck() {
        for (DuckCrossFlyPane duck : duckCrossFlyPaneDict.keySet()) {
            duck.mediaPlayer.stop();
            duck.mediaPlayer.seek(duck.mediaPlayer.getStopTime());

        }
        for (DuckVerticalFlyPane duck : duckVerticalFlyPaneDict.keySet()) {
            duck.mediaPlayer.stop();
            duck.mediaPlayer.seek(duck.mediaPlayer.getStopTime());

        }
    }
}